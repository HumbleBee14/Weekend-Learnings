"""
cute_algebra.py — A pure-Python reference implementation of the CuTe layout
algebra. Not a runtime; a teaching aid. Run with:

    pytest cute_algebra.py -v

The implementation tracks the rules in submodule 02's README:

  - Layout(shape, stride) is a function from coordinates to integers.
  - Composition A ∘ B is R(c) = A(B(c)); integral rule a:b ∘ s:d = s:(b*d).
  - Coalesce drops size-1 modes and merges adjacent contiguous modes.
  - Divide partitions a layout into (tile, rest).
  - Swizzle<BBits, MBase, SShift> applies an XOR permutation that
    eliminates shared-memory bank conflicts.

Worked examples from the README are tests below. Change the inputs, predict
the output, run the test, verify. If your prediction is wrong, re-derive
on paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import Tuple, Union

Shape = Union[int, Tuple["Shape", ...]]
Stride = Union[int, Tuple["Stride", ...]]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Layout:
    """A CuTe Layout: a (shape, stride) pair, optionally nested.

    Layout(c) returns the memory offset for coordinate c. Coordinates can
    be a tuple matching the shape, or a single integer (interpreted as a
    column-major flat coordinate).
    """

    shape: Shape
    stride: Stride

    def __post_init__(self):
        if not _same_structure(self.shape, self.stride):
            raise ValueError(
                f"shape {self.shape} and stride {self.stride} must have the "
                "same nesting structure"
            )

    # ---- core function ----

    def __call__(self, coord) -> int:
        if isinstance(coord, int):
            coord = _flat_to_nested(coord, self.shape)
        return _dot(coord, self.stride)

    def size(self) -> int:
        return _product(self.shape)

    def cosize(self) -> int:
        # max offset + 1 (assuming positive strides)
        return self(self.size() - 1) + 1 if self.size() > 0 else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _same_structure(a, b) -> bool:
    if isinstance(a, int):
        return isinstance(b, int)
    if isinstance(b, int):
        return False
    if len(a) != len(b):
        return False
    return all(_same_structure(ai, bi) for ai, bi in zip(a, b))


def _product(s: Shape) -> int:
    if isinstance(s, int):
        return s
    return reduce(lambda x, y: x * y, (_product(si) for si in s), 1)


def _dot(coord, stride) -> int:
    if isinstance(coord, int):
        assert isinstance(stride, int)
        return coord * stride
    return sum(_dot(c, d) for c, d in zip(coord, stride))


def _flat_to_nested(k: int, shape: Shape):
    """Column-major decomposition of a flat index into nested coordinates."""
    if isinstance(shape, int):
        return k
    out = []
    for s in shape:
        sz = _product(s)
        out.append(_flat_to_nested(k % sz, s))
        k //= sz
    return tuple(out)


def _flatten(s):
    if isinstance(s, int):
        return (s,)
    out = []
    for si in s:
        out.extend(_flatten(si))
    return tuple(out)


# ---------------------------------------------------------------------------
# Composition  (integral and multi-mode)
# ---------------------------------------------------------------------------


def composition(A: Layout, B: Layout) -> Layout:
    """R = A ∘ B, defined by R(c) = A(B(c)).

    Implements the integral case a:b ∘ s:d = s:(b*d), then distributes
    across multi-mode B.
    """
    # Multi-mode B: distribute.
    if isinstance(B.shape, tuple):
        sub_results = [
            composition(A, Layout(sb, db)) for sb, db in zip(B.shape, B.stride)
        ]
        return Layout(
            tuple(r.shape for r in sub_results),
            tuple(r.stride for r in sub_results),
        )

    # B is integral: B = s:d
    s, d = B.shape, B.stride

    # Integral A: a:b ∘ s:d = s : (b*d)
    if isinstance(A.shape, int):
        return Layout(s, A.stride * d)

    # Multi-mode A, integral B: walk A's flat-traversal with stride d, take s steps.
    # We do this by traversing A's modes left-to-right, consuming d at each step
    # until depleted, then taking s elements.
    # This implementation handles the canonical CuTe rules for the common
    # cases (uniform strides across modes). For pathological cases CuTe's full
    # algorithm in include/cute/algorithm/functional.hpp is the reference; this
    # implementation matches it for all examples in the README.
    flat_shape = _flatten(A.shape)
    flat_stride = _flatten(A.stride)

    # Walk: consume `d` (the multiplier on flat coord) across modes.
    result_shapes = []
    result_strides = []
    remaining = s
    current_d = d
    for fs, fd in zip(flat_shape, flat_stride):
        if remaining == 1:
            break
        # How many elements of this mode does current_d step over?
        # Mode has size fs; current_d steps it (s_mode = fs // current_d)
        if current_d >= fs:
            # Skip this mode entirely; reduce current_d
            current_d //= fs
            continue
        sub_s = min(remaining, fs // current_d)
        sub_d = fd * current_d
        result_shapes.append(sub_s)
        result_strides.append(sub_d)
        remaining //= sub_s
        current_d = 1
    if not result_shapes:
        return Layout(1, 0)
    if len(result_shapes) == 1:
        return Layout(result_shapes[0], result_strides[0])
    return Layout(tuple(result_shapes), tuple(result_strides))


# ---------------------------------------------------------------------------
# Coalesce
# ---------------------------------------------------------------------------


def coalesce(L: Layout) -> Layout:
    """Simplify a layout without changing its function:
      - drop size-1 modes
      - merge adjacent modes (s1:d1, s2:d2) when s1*d1 == d2
    """
    flat_s = _flatten(L.shape)
    flat_d = _flatten(L.stride)

    out_s = []
    out_d = []
    for s, d in zip(flat_s, flat_d):
        if s == 1:
            continue
        if out_s and out_s[-1] * out_d[-1] == d:
            out_s[-1] *= s
        else:
            out_s.append(s)
            out_d.append(d)

    if not out_s:
        return Layout(1, 0)
    if len(out_s) == 1:
        return Layout(out_s[0], out_d[0])
    return Layout(tuple(out_s), tuple(out_d))


# ---------------------------------------------------------------------------
# Complement and Divide
# ---------------------------------------------------------------------------


def complement(B: Layout, M: int) -> Layout:
    """Find R such that (B, R) covers [0, M) without overlap.

    For B = s:d, complement(B, M) = (M // (s*d)) : (s*d), repeated as needed
    to fill the space. For the 1D-case used in the README this reduces to
    a single mode.
    """
    if isinstance(B.shape, int):
        s, d = B.shape, B.stride
        cover = s * d
        rest = M // cover
        return Layout(rest, cover)
    # Multi-mode: not needed for any README example; fall back to flat.
    raise NotImplementedError("Multi-mode complement not needed for examples")


def logical_divide(A: Layout, B: Layout) -> Layout:
    """A ÷ B = A ∘ (B, complement(B, size(A))).

    Splits A into a tile (mode 0, selected by B) and the rest (mode 1).
    """
    M = A.size()
    C = complement(B, M)
    combined = Layout((B.shape, C.shape), (B.stride, C.stride))
    return composition(A, combined)


# ---------------------------------------------------------------------------
# Swizzle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Swizzle:
    """Swizzle<BBits, MBase, SShift>(addr) applies an XOR permutation.

    swizzled(a) = a XOR (((a >> SShift) & ((1 << BBits) - 1)) << MBase)

    Eliminates bank conflicts when a warp accesses a strided pattern.
    """

    BBits: int
    MBase: int
    SShift: int

    def __call__(self, addr: int) -> int:
        mask = ((1 << self.BBits) - 1) << self.MBase
        xor_bits = ((addr >> self.SShift) & ((1 << self.BBits) - 1)) << self.MBase
        return addr ^ (xor_bits & mask)


# ---------------------------------------------------------------------------
# Tests — these are the worked examples from the README.
# ---------------------------------------------------------------------------


def test_basic_indexing():
    # 4×8 row-major
    A = Layout((4, 8), (8, 1))
    assert A((2, 3)) == 19
    assert A.size() == 32

    # 4×8 column-major
    B = Layout((4, 8), (1, 4))
    assert B((2, 3)) == 14


def test_64x64_tile_nesting():
    # Row-major 64×64 BF16 storage, viewed as 4×4 grid of 16×16 tiles.
    # Inner-M strides by 64 (one row); outer-M strides by 1024 (one block-row).
    # Inner-N strides by 1; outer-N strides by 16.
    T = Layout(((16, 4), (16, 4)), ((64, 1024), (1, 16)))

    # Coord ((m_in, m_out), (n_in, n_out))
    # Element at outer (1, 2), inner (3, 5) = global row 1*16+3=19, col 2*16+5=37
    # In row-major 64×64: offset = 19*64 + 37 = 1216 + 37 = 1253
    coord = ((3, 1), (5, 2))
    assert T(coord) == 1253


def test_composition_integral():
    # a:b ∘ s:d = s : (b*d)
    A = Layout(4, 8)   # 4 elements, stride 8: maps k → 8k
    B = Layout(3, 2)   # 3 elements, stride 2: maps k → 2k
    R = composition(A, B)
    # R(k) = A(B(k)) = A(2k) = 16k
    assert R.shape == 3
    assert R.stride == 16
    for k in range(3):
        assert R(k) == A(B(k))


def test_composition_canonical():
    # (6,2):(8,2) ∘ 4:3 = (2,2) : (24,2)
    A = Layout((6, 2), (8, 2))
    B = Layout(4, 3)
    R = composition(A, B)
    # Verify by direct evaluation: R(k) = A(B(k)) for k in 0..3
    for k in range(4):
        flat_a = B(k)  # 0, 3, 6, 9
        nested = _flat_to_nested(flat_a, A.shape)
        assert R(k) == A(nested), f"k={k}: R={R(k)} A(B(k))={A(nested)}"


def test_coalesce_size_one():
    # (2,1,6):(1,6,2) coalesces by dropping the 1-mode
    # → (2,6):(1,2). Is 2*1 == 2? Yes. Merge → 12:1.
    L = Layout((2, 1, 6), (1, 6, 2))
    C = coalesce(L)
    assert C.shape == 12
    assert C.stride == 1


def test_coalesce_merge_contiguous():
    # (4, 8) : (1, 4) — adjacent modes are contiguous: 4*1 == 4. Merge → 32:1.
    L = Layout((4, 8), (1, 4))
    C = coalesce(L)
    assert C.shape == 32
    assert C.stride == 1


def test_divide_1d():
    # A = 24:1, B = 4:1
    # complement(4:1, 24) = 6:4
    # A ÷ B = ((4:1), (6:4))
    A = Layout(24, 1)
    B = Layout(4, 1)
    R = logical_divide(A, B)
    assert R.shape == (4, 6)
    assert R.stride == (1, 4)
    # Mode 0 picks elements within a tile; mode 1 picks the tile.
    # R((2, 3)) = 2*1 + 3*4 = 14 — third element of fourth tile.
    assert R((2, 3)) == 14


def test_swizzle_no_bank_conflicts():
    # 32 threads in a warp read addresses 0, row_stride, 2*row_stride, ...
    # With Swizzle<3,4,3> and a 128-byte row stride (e.g. 64 BF16),
    # the swizzled addresses should hit 32 distinct banks (mod 128 bytes).
    sw = Swizzle(3, 4, 3)
    row_stride_bytes = 128  # 64 bf16 elements
    addrs = [sw(t * row_stride_bytes // 4) for t in range(32)]
    banks = {a % 32 for a in addrs}
    assert len(banks) == 32, f"Expected 32 distinct banks, got {len(banks)}"


def test_swizzle_unswizzled_has_conflicts():
    # No swizzle: 32 threads on a 128-byte stride all hit bank 0.
    addrs = [t * 32 for t in range(32)]   # all multiples of 32 → all bank 0
    banks = {a % 32 for a in addrs}
    assert len(banks) == 1


if __name__ == "__main__":
    import pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

"""Dump StableHLO from a tiny JAX function and compare its shape to the
Inductor IR you got in Topic 02.

If JAX is not installed, the script prints a stub StableHLO program copied
from the spec so you can still see the IR shape. Install with:

    pip install --upgrade "jax[cpu]"

CPU is fine; this is about IR inspection, not performance.
"""

from __future__ import annotations


def with_jax() -> None:
    import jax
    import jax.numpy as jnp

    def block(x, w_up, w_down):
        y = jnp.matmul(x, w_up)
        y = jax.nn.silu(y)
        return jnp.matmul(y, w_down)

    rng = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(rng, 3)
    x = jax.random.normal(k1, (4, 64))
    w_up = jax.random.normal(k2, (64, 128))
    w_down = jax.random.normal(k3, (128, 64))

    lowered = jax.jit(block).lower(x, w_up, w_down)
    print("=== StableHLO (lowered) ===")
    print(lowered.as_text("stablehlo"))

    print("\n=== Compiled HLO (post-XLA optimizations, CPU backend) ===")
    compiled = lowered.compile()
    print(compiled.as_text())


def without_jax() -> None:
    print("JAX not installed; install with `pip install \"jax[cpu]\"` to dump the real IR.\n")
    print("Reference shape of a StableHLO module (from openxla.org/stablehlo/spec):\n")
    print(
        """module @block {
  func.func public @main(%arg0: tensor<4x64xf32>,
                         %arg1: tensor<64x128xf32>,
                         %arg2: tensor<128x64xf32>) -> tensor<4x64xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
           contracting_dims = [1] x [0] : (tensor<4x64xf32>, tensor<64x128xf32>) -> tensor<4x128xf32>
    %1 = stablehlo.logistic %0 : tensor<4x128xf32>
    %2 = stablehlo.multiply %0, %1 : tensor<4x128xf32>
    %3 = stablehlo.dot_general %2, %arg2,
           contracting_dims = [1] x [0] : (tensor<4x128xf32>, tensor<128x64xf32>) -> tensor<4x64xf32>
    return %3 : tensor<4x64xf32>
  }
}"""
    )


def main() -> None:
    try:
        import jax  # noqa: F401
    except Exception:
        without_jax()
        return
    with_jax()


if __name__ == "__main__":
    main()

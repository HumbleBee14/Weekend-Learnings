"""Print the O(N^2) memory wall table. No GPU needed.

Run: python memory_wall.py
"""
from __future__ import annotations


def main() -> None:
    d_head = 128
    bytes_per_elem = 2  # bf16

    print(f"{'N':>6} {'S size (MB)':>14} {'Inputs+O (MB)':>14} {'Ratio':>10}")
    for n in [512, 1024, 2048, 4096, 8192, 16384]:
        s_bytes = n * n * bytes_per_elem
        io_bytes = 4 * n * d_head * bytes_per_elem  # Q, K, V, O
        s_mb = s_bytes / 1e6
        io_mb = io_bytes / 1e6
        ratio = s_mb / io_mb
        print(f"{n:>6} {s_mb:>14.2f} {io_mb:>14.2f} {ratio:>9.1f}x")


if __name__ == "__main__":
    main()

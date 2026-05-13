"""
hello.py — your first @cute.kernel. Verifies the toolchain.

    pip install nvidia-cutlass-dsl
    python hello.py

Expected: "Hello from GPU" prints once (only thread 0 writes).

Note: API names in the cutlass.cute namespace are evolving toward 4.x
stable. If you see import errors, the wheel layout may have shifted —
check `python -c "import cutlass.cute as cute; print(dir(cute))"`.
"""

import cutlass
import cutlass.cute as cute


@cute.kernel
def hello_kernel():
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        cute.printf("Hello from GPU\n")


@cute.jit
def hello():
    cutlass.cuda.initialize_cuda_context()
    hello_kernel().launch(grid=(1, 1, 1), block=(32, 1, 1))


if __name__ == "__main__":
    hello()

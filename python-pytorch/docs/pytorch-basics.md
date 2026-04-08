# PyTorch Basics

## What is PyTorch?

PyTorch is a **numerical computation library** built around tensors, with automatic differentiation (autograd) built in. That's it. Everything else — neural networks, training loops, optimizers — is built on top of those two things.

```
PyTorch = NumPy (tensor math) + autograd (automatic gradients)
```

## What is a Tensor?

A tensor is an **n-dimensional array** — the same concept as a NumPy array, but it can run on GPU and supports automatic differentiation.

```
Scalar   → 0-dimensional tensor   → just a number:         42
Vector   → 1-dimensional tensor   → list of numbers:       [1, 2, 3]
Matrix   → 2-dimensional tensor   → table of numbers:      [[1,2],[3,4]]
Tensor   → 3+ dimensional tensor  → e.g., an image: [3, 224, 224] = 3 channels × 224×224 pixels
```

### Java analogy

```java
// Java — you'd represent this with nested arrays
float[]     vector  = {1, 2, 3};          // 1D
float[][]   matrix  = {{1,2},{3,4}};      // 2D
float[][][] tensor  = ...;                // 3D — gets ugly fast

// NumPy/PyTorch handles any dimension cleanly with one type
```

## Creating Tensors

```python
import torch

# From data directly
t = torch.tensor([1, 2, 3])               # [1, 2, 3]  — dtype inferred as int64
t = torch.tensor([1.0, 2.0, 3.0])         # [1., 2., 3.] — dtype inferred as float32

# Zeros, ones, random
t = torch.zeros(3, 4)                     # 3×4 matrix of zeros
t = torch.ones(2, 3)                      # 2×3 matrix of ones
t = torch.rand(3, 3)                      # 3×3 random values between 0 and 1
t = torch.randn(3, 3)                     # 3×3 random from standard normal distribution

# Like an existing tensor (same shape, same device)
t2 = torch.zeros_like(t)
t2 = torch.ones_like(t)
t2 = torch.rand_like(t.float())           # rand_like needs float dtype
```

## Tensor Properties

```python
t = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])

print(t.shape)      # torch.Size([2, 3])  ← 2 rows, 3 columns
print(t.dtype)      # torch.float32
print(t.device)     # cpu

# shape is used EVERYWHERE — know it cold
print(t.shape[0])   # 2 — number of rows
print(t.shape[1])   # 3 — number of columns
```

## Devices — CPU, GPU, MPS

The whole point of PyTorch over NumPy is that tensors can live on GPU:

```python
# Always write device-agnostic code:
device = (
    "cuda" if torch.cuda.is_available()        # NVIDIA GPU
    else "mps" if torch.backends.mps.is_available()  # Apple Silicon
    else "cpu"
)

t = torch.tensor([1.0, 2.0, 3.0]).to(device)  # send to GPU
t = t.to("cpu")                                 # send back to CPU

# Two tensors must be on the SAME device to operate on each other
a = torch.tensor([1.0]).to("cuda")
b = torch.tensor([2.0])                         # on CPU
c = a + b                                        # ERROR — different devices
```

## dtype — data types

```python
t = torch.tensor([1, 2, 3])            # int64 by default
t = torch.tensor([1.0, 2.0, 3.0])     # float32 by default — most common in ML

# Common dtypes
torch.float32   # default for most ML — good balance of precision and speed
torch.float16   # half precision — faster on GPU, used in mixed precision training
torch.int64     # integers — used for class labels, indices
torch.bool      # True/False — used for masks

# Cast between dtypes
t = t.float()     # → float32
t = t.half()      # → float16
t = t.long()      # → int64
```

## Indexing and Slicing

Identical to NumPy, which is similar to Python lists:

```python
t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

t[0]           # first row:        [1, 2, 3]
t[0, 1]        # row 0, col 1:     2
t[:, 1]        # all rows, col 1:  [2, 5, 8]
t[0:2, :]      # rows 0-1:         [[1,2,3],[4,5,6]]
t[t > 4]       # boolean mask:     [5, 6, 7, 8, 9]
```

## Reshaping

Shape manipulation is critical in ML — layer outputs constantly need to be reshaped:

```python
t = torch.arange(12)         # [0, 1, 2, ..., 11] — shape: [12]

t.reshape(3, 4)              # shape: [3, 4]
t.reshape(2, 2, 3)           # shape: [2, 2, 3]
t.reshape(-1, 4)             # -1 means "infer this dim" → shape: [3, 4]

# view — like reshape but must be contiguous in memory (faster, no copy)
t.view(3, 4)

# squeeze / unsqueeze — add or remove dimensions of size 1
t = torch.tensor([1.0, 2.0, 3.0])    # shape: [3]
t.unsqueeze(0)                         # shape: [1, 3] — add dim at position 0
t.unsqueeze(1)                         # shape: [3, 1] — add dim at position 1
t.squeeze()                            # removes all dims of size 1
```

## The Three Core Concepts

Everything in PyTorch builds on these three:

```
1. Tensors        — the data structure (this file)
2. Autograd       — automatic gradient computation (how training works)
3. nn.Module      — base class for all neural network layers and models
```

Learn tensors cold first. Autograd and nn.Module build on top and make no sense without it.

# Math Operations in PyTorch — From Formula to Code

## Why math matters in ML

Every operation in deep learning is math on tensors. Understanding the math tells you **why** the code does what it does:

| Operation | Where it appears in ML |
|---|---|
| Matrix multiplication | Every linear layer, attention mechanism |
| Dot product | Similarity scores, attention weights |
| Element-wise multiplication | Gating mechanisms (LSTM, GRU), loss scaling |
| Transpose | Reshaping for matmul, attention |
| Sum / Mean | Loss functions, pooling layers |
| Softmax | Probability distributions, attention weights |
| Norm | Regularization, gradient clipping |

---

## 1. Element-wise Operations

**Math:** apply operation to each corresponding element

```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]

A + B = [[1+5, 2+6], = [[6,  8],
          [3+7, 4+8]]    [10, 12]]
```

```python
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

a + b                  # addition
a - b                  # subtraction
a * b                  # element-wise multiply (NOT matrix multiply — common mistake)
a / b                  # element-wise divide
a ** 2                 # element-wise square
torch.sqrt(a)          # element-wise square root
```

**ML relevance:** activation functions (ReLU, sigmoid) apply element-wise. Loss scaling applies element-wise. Any mask (set certain values to 0) is element-wise multiply.

---

## 2. Dot Product

**Math:** multiply corresponding elements, sum them all up

```
a = [1, 2, 3]
b = [4, 5, 6]

a · b = (1×4) + (2×5) + (3×6) = 4 + 10 + 18 = 32
```

```python
a = torch.tensor([1., 2., 3.])
b = torch.tensor([4., 5., 6.])

torch.dot(a, b)        # 32.0 — only for 1D tensors
```

**ML relevance:** measures **similarity** between two vectors. If both are unit vectors (length 1), dot product = cosine similarity. Used in attention to score how relevant each token is to the query.

```
query · key = similarity score
→ high dot product = vectors point in same direction = relevant
→ low/negative     = vectors point away             = irrelevant
```

---

## 3. Matrix Multiplication (matmul)

**Math:** row of A dot product with column of B

```
A = [[1, 2],    B = [[5, 6],
     [3, 4]]         [7, 8]]

C = A @ B:
C[0,0] = row0_A · col0_B = (1×5) + (2×7) = 19
C[0,1] = row0_A · col1_B = (1×6) + (2×8) = 22
C[1,0] = row1_A · col0_B = (3×5) + (4×7) = 43
C[1,1] = row1_A · col1_B = (3×6) + (4×8) = 50

C = [[19, 22],
     [43, 50]]

Rule: A is (m×n), B is (n×p) → C is (m×p)
             ↑_______↑  inner dimensions MUST match
```

```python
a = torch.tensor([[1., 2.], [3., 4.]])    # shape: [2, 2]
b = torch.tensor([[5., 6.], [7., 8.]])    # shape: [2, 2]

a @ b                      # matrix multiply — most common syntax
torch.matmul(a, b)         # same thing
torch.mm(a, b)             # same, only for 2D

# Batched matmul — multiplies each matrix in a batch
a = torch.randn(32, 10, 512)    # batch of 32, each matrix is (10×512)
b = torch.randn(32, 512, 64)    # batch of 32, each matrix is (512×64)
c = torch.matmul(a, b)          # shape: [32, 10, 64]
# @ handles this too — bmm() also exists for explicit batched matmul
```

**ML relevance:** this IS the linear layer. `y = xW + b` — the `xW` is a matmul:

```python
x = torch.randn(32, 784)      # 32 examples, each with 784 features (e.g., MNIST)
W = torch.randn(784, 256)     # weight matrix of a linear layer
b = torch.randn(256)          # bias

y = x @ W + b                 # shape: [32, 256]
# This is literally what nn.Linear does internally
```

In the attention mechanism:
```
Q @ K.T / sqrt(d_k) → attention scores
scores @ V           → attention output
```
It's all matmul.

---

## 4. Transpose

**Math:** flip rows and columns

```
A = [[1, 2, 3],     A.T = [[1, 4],
     [4, 5, 6]]             [2, 5],
                             [3, 6]]

A is (2×3) → A.T is (3×2)
```

```python
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])   # shape: [2, 3]

a.T                    # shape: [3, 2]
a.transpose(0, 1)      # same — transpose dims 0 and 1
a.permute(1, 0)        # same — permute is the general version for any dims

# permute for 3D+ tensors — e.g., images
img = torch.randn(224, 224, 3)     # H × W × C (common in OpenCV)
img = img.permute(2, 0, 1)         # → C × H × W (PyTorch expects this format)
```

**ML relevance:** in attention, `K.T` (transpose of keys) is needed to compute `Q @ K.T`. Also used constantly to align dimensions before matmul.

---

## 5. Reduction Operations — Sum, Mean, Max

**Math:** reduce a dimension to a single value

```
A = [[1, 2, 3],
     [4, 5, 6]]

sum(all)       = 21
sum(dim=0)     = [5, 7, 9]    ← sum each column (across rows)
sum(dim=1)     = [6, 15]      ← sum each row (across columns)
mean(dim=1)    = [2., 5.]     ← mean of each row
```

```python
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])

a.sum()                # scalar: 21
a.sum(dim=0)           # shape: [3] — collapse rows
a.sum(dim=1)           # shape: [2] — collapse columns
a.mean(dim=1)          # shape: [2]
a.max()                # scalar max
a.max(dim=1)           # returns (values, indices) along dim
a.argmax(dim=1)        # index of max per row
```

**ML relevance:**
- **Mean loss** across a batch: `loss.mean()`
- **Softmax** uses sum to normalize: `exp(x) / exp(x).sum()`
- **Global average pooling** in CNNs: `x.mean(dim=[2,3])` collapses spatial dims
- **Argmax** at inference: "which class has the highest score?"

---

## 6. Broadcasting

**Math:** automatically expand smaller tensor to match larger tensor's shape

```
A = [[1, 2, 3],    b = [10, 20, 30]
     [4, 5, 6]]

A + b → b is broadcast across rows:
     = [[1+10, 2+20, 3+30],
        [4+10, 5+20, 6+30]]
     = [[11, 22, 33],
        [14, 25, 36]]
```

```python
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])   # shape: [2, 3]
b = torch.tensor([10., 20., 30.])                  # shape: [3]

a + b   # b broadcast to [2, 3] → works
```

**ML relevance:** adding bias in a linear layer — bias is shape `[256]`, output is `[32, 256]`. Broadcasting adds bias to every example in the batch without manually repeating it.

---

## 7. Softmax

**Math:** convert raw scores (logits) to probabilities that sum to 1

```
x = [2.0, 1.0, 0.1]

softmax(x_i) = exp(x_i) / Σ exp(x_j)

exp([2.0, 1.0, 0.1]) = [7.389, 2.718, 1.105]
sum                   = 11.212

softmax = [7.389/11.212, 2.718/11.212, 1.105/11.212]
        = [0.659, 0.242, 0.099]   ← sum = 1.0 ✓
```

```python
x = torch.tensor([2.0, 1.0, 0.1])

torch.softmax(x, dim=0)           # [0.659, 0.242, 0.099]

# For batched logits (common):
logits = torch.randn(32, 10)       # 32 examples, 10 class scores
probs = torch.softmax(logits, dim=1)   # dim=1 — softmax across classes per example
```

**ML relevance:** the final layer of any classifier. Also used in **attention** — attention scores are softmaxed so attention weights sum to 1 (a probability distribution over tokens).

---

## 8. Norm

**Math:** magnitude (length) of a vector

```
L2 norm (Euclidean):  ||x|| = sqrt(x₁² + x₂² + ... + xₙ²)

x = [3, 4]
||x|| = sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
```

```python
x = torch.tensor([3., 4.])

torch.norm(x)              # 5.0  (L2 norm by default)
torch.norm(x, p=1)         # L1 norm: |3| + |4| = 7
torch.norm(x, p=2)         # L2 norm: 5.0

# Matrix norm
W = torch.randn(256, 512)
torch.norm(W)              # Frobenius norm (default for matrices)
```

**ML relevance:**
- **Gradient clipping** — if `||gradients|| > threshold`, scale them down. Prevents exploding gradients in LLM training
- **L2 regularization (weight decay)** — adds `||W||²` to loss to prevent overfitting
- **Normalizing vectors** — divide by norm to get unit vector (used in cosine similarity, embedding spaces)

---

## Quick Reference — Operation → PyTorch

| Math | Symbol | PyTorch |
|------|--------|---------|
| Element-wise multiply | A ⊙ B | `a * b` |
| Matrix multiply | AB | `a @ b` or `torch.matmul(a, b)` |
| Dot product | a · b | `torch.dot(a, b)` |
| Transpose | Aᵀ | `a.T` or `a.transpose(0,1)` |
| Sum all | Σ | `a.sum()` |
| Sum along axis | Σᵢ | `a.sum(dim=i)` |
| Mean | μ | `a.mean()` |
| Softmax | softmax(x) | `torch.softmax(x, dim=d)` |
| L2 norm | \|\|x\|\| | `torch.norm(x)` |
| Square root | √x | `torch.sqrt(x)` |
| Exponential | eˣ | `torch.exp(x)` |
| Log | ln(x) | `torch.log(x)` |

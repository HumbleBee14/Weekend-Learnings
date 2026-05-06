# Testing for ML systems

A meta-topic referenced from each level's projects. ML systems testing is different from web app testing — same fundamentals, different failure modes, different assertions.

This doc is the place those patterns live so each level can point at it instead of repeating itself.

## Why ML systems testing is different

A web app test asserts: "GET /users/42 returns this JSON." A correct response either matches or doesn't. Binary.

An ML system test has fuzzier truth:
- Floating-point math is non-deterministic across hardware. The "right" answer is "within tolerance ε."
- Sampling adds randomness. Same prompt + same seed should produce the same output, but a different seed legitimately produces something different.
- Performance is part of correctness. A kernel that returns the right answer at 10% of expected throughput is broken.
- Some bugs only show under load. Race conditions, queue overflow, OOM, head-of-line blocking — none show up in unit tests.

Each of these requires different testing patterns.

## The seven testing layers for ML systems

```
                                     fast        slow
   1. Type / schema tests           <100ms       per file
   2. Numerical correctness         <1s          per kernel
   3. Tolerance-bounded equivalence ~seconds     per layer
   4. Property-based / fuzz         ~minutes     per component
   5. Integration (golden files)    ~minutes     per request path
   6. Load / chaos                  ~hours       per environment
   7. Quality regression            ~hours       per release
```

Each layer catches different bugs. Skipping any layer leaves a class of bug uncaught.

## Layer 1 — Type and schema tests

Pydantic schema validation, JSON contract tests for HTTP boundaries, type checking with `mypy` or `pyright`. Run on every commit, in CI, in <1 minute total.

Already in the curriculum: Level 1 capstone has `tests/test_schemas.py`. The pattern:

```python
def test_generate_request_validates_max_tokens():
    with pytest.raises(ValueError):
        GenerateRequest(prompt="x", max_tokens=0)
```

Cheap, fast, catches the dumbest bugs. Have these for every Pydantic model, every API route, every config object.

## Layer 2 — Numerical correctness

For kernels and ops: assert the output matches a reference implementation within tolerance.

```python
def test_fused_rmsnorm_matches_torch():
    x = torch.randn(64, 4096, device="cuda", dtype=torch.float16)
    weight = torch.randn(4096, device="cuda", dtype=torch.float16)
    
    out_fused = my_fused_rmsnorm(x, weight, eps=1e-6)
    out_ref   = F.rms_norm(x, normalized_shape=(4096,), weight=weight, eps=1e-6)
    
    torch.testing.assert_close(out_fused, out_ref, rtol=1e-2, atol=1e-3)
```

**Tolerance choice matters.** Default tolerances assume FP32. For FP16, atol ~1e-3 is sane. For FP8, ~1e-2. For BF16, ~5e-3. Tighter than that and you'll get false positives from legitimate floating-point reordering. Looser and you'll miss real bugs.

**Test the edges.**
- Non-power-of-2 sizes (`N=1000`, `N=4097`)
- Very small inputs (`N=1`)
- Inputs with infinities, NaNs, very large values, very small values
- Zero-length sequences (causal masking corner case)

Many kernel bugs hide in the "boundary tile" — the last tile that's not fully filled. If your test only uses `N=4096` (cleanly tiled), you'll never hit it.

## Layer 3 — Tolerance-bounded equivalence

A level up from kernel-correctness: end-to-end equivalence of two implementations of the same model layer or full forward pass.

```python
def test_my_model_matches_hf_reference():
    hf_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    my_model = MyImplementation.from_hf(hf_model)
    
    input_ids = torch.tensor([[1, 2, 3, 4]])
    
    with torch.no_grad():
        out_hf = hf_model(input_ids).logits
        out_mine = my_model(input_ids).logits
    
    torch.testing.assert_close(out_hf, out_mine, rtol=5e-2, atol=5e-2)
```

Wider tolerance than kernel-level because errors compound through layers. Use this when you build something model-shaped (your `mini-vllm` from Level 4 against HF reference).

## Layer 4 — Property-based / fuzz testing

For batchers, schedulers, queue management — anything where the input space is too large to enumerate. Use `hypothesis` (Python's de facto property-based testing library).

```python
from hypothesis import given, strategies as st

@given(
    requests=st.lists(
        st.tuples(
            st.text(min_size=1, max_size=100),    # prompt
            st.integers(1, 100),                   # max_tokens
        ),
        min_size=1, max_size=20,
    )
)
def test_batcher_never_drops_requests(requests):
    batcher = Batcher(max_batch_size=8)
    futures = [batcher.submit(r) for r in requests]
    results = [f.result(timeout=60) for f in futures]
    assert len(results) == len(requests)
    assert all(r is not None for r in results)
```

Hypothesis generates hundreds of inputs, finds the smallest one that breaks the property, and reports it. Catches: edge cases you didn't think of, race conditions if the batcher is concurrent, sizing bugs.

**Where it shines for ML systems:**
- Continuous batching (does it preserve request order? does it correctly handle requests joining mid-decode?)
- KV cache eviction (does the cache stay below capacity under any access pattern?)
- Routing logic (does prefix matching produce the same output regardless of request ordering?)

## Layer 5 — Integration tests with golden files

For request paths: send a request, compare full response to a stored "golden" output.

```python
def test_chat_completion_smoke():
    response = client.post("/v1/chat/completions", json={
        "model": "qwen2.5-0.5b",
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0,           # greedy → deterministic
        "max_tokens": 10,
    })
    
    expected = json.loads(Path("tests/golden/chat_smoke.json").read_text())
    
    # Compare the deterministic parts; ignore timestamps and request IDs
    actual = response.json()
    actual.pop("created", None)
    actual.pop("id", None)
    assert actual == expected
```

**Greedy sampling makes this work.** With `temperature=0`, decoding is deterministic given the same model + same hardware. Different hardware can change FP arithmetic order, so golden files are only valid for one hardware target. CI for ML systems should pin the GPU type.

For non-deterministic scenarios (sampling), test invariants instead: "the response has at least 1 token," "the response's `finish_reason` is one of {stop, length, error}."

## Layer 6 — Load and chaos

Test the system under real load and under failure injection. Already in the curriculum:
- Level 1 Topic 05 (Locust load tests, latency CDFs)
- Level 6 Topic 12 (failure injection — kill a node mid-training, recover via NCCL Comm Shrink)
- Level 7 (the break-it list — cold start under load, scheduler swap, regression gate, traffic skew, queue threshold, cancellation propagation)

The pattern: don't just test the happy path. Inject the failures *you expect to happen in production* and assert the system recovers.

```python
def test_server_recovers_from_oom():
    # Send a giant request that forces OOM
    requests.post("/generate", json={"prompt": "x" * 10**6, "max_tokens": 100})
    
    # Subsequent requests should still work
    r = requests.post("/generate", json={"prompt": "Hi", "max_tokens": 10})
    assert r.status_code == 200
    
    # Health endpoint should report ok (not stuck in degraded state)
    h = requests.get("/health").json()
    assert h["status"] == "ok"
```

## Layer 7 — Quality regression

The ML-specific layer that doesn't exist in regular web app testing.

If you change your KV cache, your fused kernel, your quantization recipe — does the model still produce *correct outputs* across a benchmark? Run `lm-eval-harness` on a small subset (MMLU 5-shot, ARC, GSM8K-100). Compare to the previous build's score. Fail the build if the score drops more than 0.5% absolute.

```python
def test_no_quality_regression():
    score = run_lm_eval_subset(model_path="builds/current/", tasks=["mmlu", "arc_easy"])
    baseline = json.loads(Path("baseline_scores.json").read_text())
    
    for task, current_score in score.items():
        baseline_score = baseline[task]
        regression = baseline_score - current_score
        assert regression < 0.005, f"{task} regressed by {regression:.3f}"
```

Slow (hours, not seconds). Run on every release, not every commit.

## CI patterns for GPU code

The hard part: GPU CI is expensive. Patterns:

1. **Tier the test suite.**
   - **Tier 1 (every commit, <2 min)**: schema, numerical correctness on small inputs, lints. CPU only.
   - **Tier 2 (every PR, ~10 min)**: GPU smoke tests, integration tests with one model, layer 5.
   - **Tier 3 (nightly, ~hours)**: load tests, quality regression, cross-hardware.

2. **Mock the GPU when possible.** PyTorch on CPU can run small models for layer 5. Triton has a CPU emulation backend (`TRITON_INTERPRET=1`) for kernel logic correctness.

3. **Pin the hardware.** GPU CI on a self-hosted runner with a known GPU model. Don't trust "any A100" — kernel autotune picks differently on different cards.

4. **Capture artifacts.** When a test fails, automatically capture: the input, the actual output, the expected output, NSight trace if it's a perf regression. Debugging GPU code without these artifacts is awful.

## What this means for each level

- **Level 1**: layers 1, 2, 5 — schema tests, kernel correctness, integration smoke tests.
- **Level 2**: layer 2 (kernel correctness with tolerance) is the main one. `compute-sanitizer` catches most race conditions for free.
- **Level 3**: this whole doc applies — profiling output is a test artifact too.
- **Level 4**: layers 2, 3, 7 — your paged KV implementation must match a reference; quality regression on quantized models.
- **Level 5**: layer 5 (the bake-off itself is integration testing) + layer 7 (quality regression across engines).
- **Level 6**: layer 6 (failure injection) is the headline; layer 4 (property-based for collective primitives) helps.
- **Level 7**: layer 6 (load + chaos) is the project. Layer 4 (property-based for routers) is also valuable.

## Tools

| Tool | Use |
|---|---|
| `pytest` | Test runner |
| `pytest-asyncio` | Async test support |
| `hypothesis` | Property-based / fuzz testing |
| `torch.testing.assert_close` | Tolerance-bounded tensor equality |
| `compute-sanitizer` | CUDA race / OOB detection (replaces deprecated `cuda-memcheck`) |
| `lm-eval-harness` | Model quality benchmarks for layer 7 |
| `locust` | Load testing (Level 1 Topic 05) |
| `pytest-benchmark` | Performance regression tests |
| `cuda-gdb` | Kernel debugging when things hang |

## References

- `torch.testing` — https://pytorch.org/docs/stable/testing.html
- Hypothesis — https://hypothesis.readthedocs.io/
- Compute Sanitizer — https://docs.nvidia.com/compute-sanitizer/
- lm-evaluation-harness — https://github.com/EleutherAI/lm-evaluation-harness

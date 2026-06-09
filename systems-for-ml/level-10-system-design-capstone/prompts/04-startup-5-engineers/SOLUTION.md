# Prompt 04 — Worked Solution

> Open this only after attempting the prompt cold. This is one defensible design, not the only one.

## 1. Clarifying questions (the first 3 minutes)

A senior candidate asks these *before* drawing anything. They scope the design and signal that you've done this before:

1. **Token distribution.** Mean input / output tokens? P99 input / output? (Affects KV memory + decode-vs-prefill split.)
2. **Tenancy & isolation.** One model serves all customers, or per-customer fine-tunes? Multi-LoRA or full base swaps? (Affects whether multi-LoRA serving is the answer.)
3. **Quality target.** Is `lm-eval-harness` part of the deploy gate, or "ship it if no smoke"? (Affects eval-pipeline complexity.)
4. **Eng-time budget.** Is the CTO willing to fund 1 person full-time on infra, or is this "infra is whoever has the cycles this sprint"? (This is the substrate-choice question. 5 engineers + 1 full-time-on-infra = different answer from 5 engineers + nobody dedicated.)
5. **Compliance posture.** Is the customer's data leaving the platform OK? GDPR / SOC2 timeline? (Affects multi-cloud, region pinning.)

**Reasonable assumptions to bake in if the interviewer waves off:**
- 800 input tokens mean, 300 output tokens mean
- 2000 / 1200 p99 in / out
- One base model + 2–3 LoRA variants (not 500)
- "Ship it if no smoke" with a regression gate before promote
- No dedicated infra person — eng-time is precious
- US-only for now, SOC2 next year

## 2. The right answer in one sentence

**Modal (or RunPod Serverless) for the first 9–12 months. Migrate to owned K8s only when steady-state QPS justifies the engineering investment — likely the 5× growth event.**

Why this is the right answer for *this* startup:
- 5 engineers + no dedicated infra → owning Kubernetes is a poor allocation of attention
- 20× peak-to-trough ratio (40 / 2) → scale-to-zero saves real money
- 2–3 new fine-tunes / month → fast deploy iteration matters more than $/Mtok
- p95 < 1.5s warm → Modal's ~12s cold start is fine if warm pool covers business hours
- 5× growth is uncertain → don't lock in a 6-month K8s migration on a maybe

A senior candidate **names this tradeoff explicitly** rather than reflexively reaching for K8s.

## 3. The architecture (whiteboard)

```
                   Internet
                      │
                      ▼
              ┌────────────────┐
              │   Cloudflare   │   L7 — TLS, WAF, rate-limit per-tenant
              │   (or just     │   $5/mo
              │   Modal's      │
              │   gateway)     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │   Modal App    │   ─ scale-to-zero in night hours
              │                │   ─ keep_warm=2 during US business hours
              │  ┌──────────┐  │   ─ GPU memory snapshotting (Modal owns)
              │  │ vLLM     │  │   ─ vLLM with LoRA hot-swap enabled
              │  │ + LoRAs  │  │   ─ engine version pinned per deploy
              │  └──────────┘  │
              └───────┬────────┘
                      │ (metrics + logs)
                      ▼
              ┌────────────────┐
              │  Observability │   ─ Prometheus → Grafana Cloud free tier
              │  + Cost board  │   ─ OTel GenAI semconv
              │                │   ─ Slack alert on p95 > 1.5s or 5×$/Mtok deviation
              └────────────────┘

              ┌────────────────┐
              │  Eval pipeline │   ─ GitHub Action — on every fine-tune PR:
              │  (GitHub       │     1. lm-eval-harness on 5 task subset
              │   Action)      │     2. custom "support-quality" eval on 100 prompts
              │                │     3. promote-to-prod gate if regression < 2%
              └────────────────┘
```

### The five-box mapping
- **Gateway:** Cloudflare in front of Modal (or just Modal's built-in URL — startup-pragmatic)
- **Router:** Modal's load balancer (free, built-in) — no KV-cache-aware routing needed at 40 QPS
- **Scheduler:** vLLM's continuous batcher does it inside each container
- **Worker:** Modal containers running vLLM with multi-LoRA enabled
- **Control plane:** GitHub Actions for eval + deploy + promote

**The senior signal here is what's missing:** no Kubernetes, no KEDA, no custom router, no Prometheus operator deployment, no LMCache. At 40 QPS, all of those add ops burden without proportional value.

## 4. The capacity math

```
Mean throughput required:
  input  = 40 QPS × 800 tok = 32K tok/s
  output = 40 QPS × 300 tok = 12K tok/s

vLLM 13B on a single L40S (rough numbers, your bake-off should verify):
  prefill   ≈ 6–8K tok/s/GPU  (FP16; L40S caps at ~362 BF16 TFLOPS, so the
                               13B prefill ceiling is ~14K tok/s at 100% MFU,
                               ~6–8K at a realistic 40–55% MFU. FP8 weights
                               push this toward ~20K — quote the FP8 number
                               only if you're actually serving FP8.)
  decode    ≈ 950 tok/s/GPU   (bandwidth-bound; L40S 864 GB/s)
  concurrent decode slots ≈ 48 (short-context max; at the full 1100-tok
                               request size, KV budget fits only ~8–9
                               simultaneous full-length sequences)

GPUs needed (mean):
  prefill   = 32K / (7K × 0.70)  = 6.5 → 7
  decode    = 12K / (950 × 0.70) = 18.05 → 19 (round UP — it's a capacity floor)
  concurrency = (40 × ~16s lifetime) / 48 = 13 → 14
  ← decode binds HERE (output-heavy chat shape, per-GPU decode ~7–23× below
    prefill). NOT a universal law: for long-input/short-output workloads
    (RAG, summarization, classification) prefill dominates — that asymmetry
    is the whole premise of disaggregated prefill/decode serving.

Binding: 19 GPUs at mean. With Modal's per-second billing,
ACTUAL GPU-seconds billed = total tokens / engine_throughput =
   (40 × 1100 × 3600 × 8hr_business) / decode_tok/s ≈ 1.5M GPU-seconds/day at peak

But — and this is the key — at night you pay zero. Average daily fleet
is closer to 6 GPUs equivalent, not 19.
```

### Cost comparison

```
Option                          Eng-hours setup   $/Mtok blended  Monthly bill at 40 QPS peak
───────────────────────────────────────────────────────────────────────────────────────────
Modal (recommended)             ~8h               $1.45           $4,200
RunPod Serverless               ~16h              $1.10           $3,400
K8s + vLLM (4 L40S + KEDA)      ~80h              $0.85           $2,800 + 0.3 FTE forever
Replicate (only 1 base model)   ~2h               $2.10           $5,800

K8s wins on $$ but burns 80h of setup + 0.3 FTE of someone's time forever.
For a 5-eng startup, the $1,400/month Modal premium buys you back 0.3 FTE.
```

**The senior signal: do the math, then defend the choice even when it's not the cheapest line.** Eng-time has a $/hour rate. At a Series A startup it's probably $150-200/hour fully loaded. 0.3 FTE × $300K = $90K/year of attention saved. Modal's $1,400/month markup is $16,800/year. Net: Modal wins by ~$70K/year of recovered eng-time.

## 5. Multi-LoRA on Modal

This is the gotcha that distinguishes a Level 5/L7 grad from a bluffer:

```python
# Modal app — vLLM with multi-LoRA enabled
@app.cls(
    gpu="L40S",
    container_idle_timeout=300,    # ← scale-to-zero after 5min of no traffic
    keep_warm=2,                    # ← warm pool during business hours
    timeout=300,
)
class InferenceServer:
    @modal.enter()
    def load(self):
        from vllm import LLM, EngineArgs
        self.llm = LLM(
            model="Qwen/Qwen2.5-13B-Instruct",
            enable_lora=True,           # ← the magic flag
            max_loras=4,
            max_lora_rank=32,
        )

    @modal.method()
    def generate(self, prompt: str, lora_name: str | None = None):
        # vLLM hot-swaps the LoRA per-request, no container restart
        lora_request = LoRARequest(lora_name, ...) if lora_name else None
        return self.llm.generate(prompt, lora_request=lora_request)
```

You ship a new fine-tune by: train LoRA → push to HF Hub → update the LoRA registry in your eval pipeline → eval-gate passes → register in Modal config → next request uses it. No container rebuild. No K8s rolling update. **This is the workflow that justifies the substrate choice for a 5-eng startup shipping 2–3 fine-tunes a month.**

## 6. The break-it list (what fails, what you do)

Don't skip this section — it's the highest-leverage 5 minutes of the interview.

| Failure | What happens | Your mitigation |
|---|---|---|
| Modal region outage | All traffic dies | Pre-deploy parallel RunPod handler; manual DNS failover; or accept 1h downtime at this stage |
| Key customer 5× burst Monday morning | Cold-start avalanche → p95 spikes | Pre-warm by scheduled cron at 8am ET; raise `keep_warm` to 5 the night before known launches |
| New fine-tune regresses on edge cases | Customer complaints | Regression gate in eval pipeline blocks promote; canary deploy at 5% traffic for 24h before full rollout |
| vLLM version bump breaks multi-LoRA | All serving broken | Pin engine version per deploy; staging environment hits same Modal image first |
| Modal raises prices 30% (real risk) | $/Mtok spikes | Have RunPod handler 1-day-from-prod; migration is portable because we wrote it that way |
| 5× growth materializes | Cost crosses K8s breakeven | Migrate. We were ready. |

## 7. What changes at 10× scale

The most-asked follow-up question in every senior-eng interview. Have a clear answer:

```
At 400 QPS sustained (10× the 40 QPS peak):
  - decode-bound GPU count: 180 GPUs equivalent
  - Modal $/Mtok premium becomes $14K/month over K8s
  - Now worth a 6-month K8s migration:
       * vLLM Production Stack as router
       * KEDA on vllm:num_requests_waiting
       * Owned H100 reservation (committed-use discount)
       * Prometheus + Grafana proper deployment
       * One full-time platform engineer hired
  - At this stage: also disaggregated prefill/decode (Topic 08 + 09)
  - LMCache for cross-replica KV (Topic 12)
  - Maybe migrate to FP8 quantization (Topic 02) for another 30% $/Mtok cut
```

**The trick to answering "what changes at 10× scale" is to name the *axis* of change,** not just "we'd use more servers." Axes: substrate (PaaS → K8s), cost-optimization (FP8, KV reuse), reliability (single-region → multi-region), team-shape (5 eng → 1 dedicated platform eng).

## 8. The 30-second summary you give the panel

> "For 40 QPS peak on a 13B with 20× peak-to-trough and 2-3 fine-tunes per month, I'd ship on Modal with vLLM multi-LoRA enabled. Eng-time is the dominant cost at a Series A, and Modal's snapshotting beats anything we'd build ourselves in under a quarter. At 10× growth or ~200 QPS sustained, the math flips and we migrate to owned K8s with vLLM Production Stack — that's a 6-month project I'd start when our 90-day cost forecast crosses the breakeven. Multi-LoRA hot-swap stays the deploy story either way."

If you can deliver that in 30 seconds at the end of the 45 minutes, you passed. Most candidates can't.

## What this prompt is really testing

- **Substrate judgment** (Topic 11b) — most candidates default to Kubernetes
- **Cost-aware design** (Topic 13 + CAPACITY-PLANNING.md) — most candidates ignore eng-time as a cost
- **Multi-LoRA mechanics** (Level 5 Topic 10) — separates the bluffers
- **Migration thinking** — "what changes at 10× scale" is the seniority signal
- **What you don't build** — naming things you intentionally skip (KEDA, custom router, LMCache) at this scale is a strong signal

## References

- [Topic 11b — serverless GPU substrates](../../../level-7-ml-platform/11b-serverless-gpu-substrates/)
- [Topic 13 — cost-economics](../../../level-7-ml-platform/13-cost-economics/) and its [CAPACITY-PLANNING.md](../../../level-7-ml-platform/13-cost-economics/CAPACITY-PLANNING.md)
- [Topic 10 — multi-LoRA serving](../../../level-5-production-engines/10-multi-lora-serving/)
- Kiely Ch 7 §7.2–7.3 (autoscaling, multi-cloud) — the practitioner framing
- [Modal docs on scale-to-zero](https://modal.com/docs/guide/scale)

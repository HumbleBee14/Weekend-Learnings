# Prompt 04 — Rubric

What an interviewer grades on for *this specific* prompt.

## Strong signals (3 → 4)

- **Names substrate choice as a decision** with explicit tradeoffs (eng-time vs $/Mtok vs lock-in)
- Picks **Modal or RunPod**, not K8s — and defends *why* given 5 engineers, no dedicated infra, 20× peak-to-trough
- Does the **multi-LoRA hot-swap** math correctly — knows it's a vLLM flag, not a fresh container per LoRA
- Quantifies **eng-time as a cost** in $/year terms ("0.3 FTE × $300K = $90K/year saved")
- Names the **10× migration trigger** explicitly — knows when to leave the PaaS
- Mentions **regression gate** in the eval pipeline for the 2-3-fine-tunes-per-month cadence
- Has a story for **Monday morning burst** (pre-warm, scheduled cron, raise keep_warm)
- **Excludes** components correctly — explicitly skips KEDA, LMCache, custom router at this QPS

## Solid signals (passing — a 3)

- Reaches for vLLM as the engine (not "OpenAI API" or "we'll figure it out")
- Knows the 5-box architecture and labels each box
- Does *some* GPU sizing math even if rough
- Mentions cold-start as a concern with at least one mitigation
- Has *some* observability story

## Anti-signals (instant downgrade to 2)

- "We'd deploy on Kubernetes" — without naming why over serverless at 40 QPS for a 5-eng startup
- "We'd build a custom router" — at 40 QPS this is over-engineering
- No numbers anywhere
- Forgot multi-LoRA entirely (the prompt mentions 2–3 fine-tunes/month — they're testing this)
- "We'd just use OpenAI API" — the prompt says "fine-tuned 13B"; they want you to serve it
- Calls Modal "expensive" without quantifying eng-time savings
- Designs a system that can't roll back a bad fine-tune

## Follow-up questions to expect

1. *"What if Modal goes down?"* → Has multi-cloud failover answer (RunPod warm)
2. *"What about a customer who needs SOC2?"* → Knows Modal has SOC2, region pinning available
3. *"How do you know your fine-tune didn't regress?"* → Eval pipeline gate, canary deploy, lm-eval-harness
4. *"At 10× growth, what changes?"* → Migration trigger, K8s + vLLM Production Stack, FP8, dedicated platform eng
5. *"Walk me through the cost math again."* → Should re-derive on the spot without flinching

## What this prompt is really testing

The interview is **substrate judgment + cost-aware design + restraint**. Most senior candidates reflexively reach for Kubernetes because it's what they used at their last job. The 4-signal is recognizing that *the right substrate is workload-shape-dependent and team-shape-dependent*, not engineer-preference-dependent.

## Time budget for the candidate

| Minutes | What you should be doing |
|---|---|
| 0–3 | Clarifying questions (3-5 of them) |
| 3–8 | Architecture diagram on the board |
| 8–18 | Substrate choice + defense |
| 18–28 | Cost math + multi-LoRA mechanics |
| 28–35 | Break-it list (failure modes) |
| 35–42 | 10× scale story |
| 42–45 | 30-second summary |

A senior candidate finishes in ~40 minutes and uses the last 5 for the follow-up Q&A.

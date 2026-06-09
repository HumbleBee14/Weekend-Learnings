# 11b — Serverless GPU Substrates (Modal / RunPod / Replicate / Beam)

> Position in the level: Topic 11 taught you to *own* the cold-start problem on Kubernetes. This topic teaches you what it looks like when you *delegate* it to a PaaS — and where that delegation breaks.

## Files

- `CONCEPTS.md` — what serverless GPU actually is (under the hood), the four-way 2026 comparison (Modal / RunPod / Replicate / Beam), where it wins vs where it loses, the migration decision tree.
- `modal_app.py` — same `mini-serve`-shape inference, deployed on Modal in ~50 lines. Compare to your K8s deployment.
- `runpod_handler.py` — same on RunPod Serverless via their handler API.
- `compare_substrates.md` — worked example: same 7B model, same load, K8s+vLLM (Topic 10) vs Modal vs RunPod. Cold start, warm latency, $/Mtok, ops burden.

## Why this topic exists

In 2026, serverless GPU is no longer a fringe deployment pattern — Modal raised $87M Series B at $1.1B (Sept 2025), Cloudflare acquired Replicate (Nov 2025), Modular acquired BentoML (Feb 2026). Three reasons platform teams now have to know this substrate:

1. **It's where most teams ship first.** A startup with one engineer and ten model variants isn't going to operate Kubernetes. They'll ship on Modal or RunPod and migrate to owned infra when economics force it.
2. **It changes the cost story.** Pay-per-second with no warm pool means $/Mtok looks dramatically different from Topic 13's matrix at low/spiky QPS. At high steady-state QPS the math inverts — owned infra wins by ~2-3×.
3. **It changes the cold-start story.** Topic 11's seven phases still happen — but the PaaS owns them. Modal's GPU memory snapshotting cuts cold start from ~70s to ~12s for free; you can't beat that on your own K8s cluster without significant engineering.

## What serverless GPU *is*, mechanically

```
   Owned K8s (Topics 10-11)            Serverless GPU (this topic)
   ──────────────────────              ────────────────────────────
   you own the cluster                 PaaS owns the cluster
   you provision N GPU nodes           PaaS multiplexes their fleet
   you run KEDA + Prometheus           PaaS handles autoscale
   you manage cold starts              PaaS handles cold starts (snapshotting)
   you pay per GPU-hour                you pay per GPU-second (active only)
   you can scale to zero (impractical) PaaS scale-to-zero is the default
   your warm pool is a $$ line item    no warm pool to pay for
   markup: ~10-15% over raw GPU $/hr   markup: ~30-50% over raw GPU $/hr
   ops burden: high                    ops burden: very low
   freedom: total                      freedom: limited to PaaS abstractions
```

The platform achieves "truly serverless" via three tricks: (1) **GPU memory snapshotting** — the PaaS takes a checkpoint of GPU state after weight load + warmup, restores it on cold-start (Modal's secret sauce, ~6× cold-start reduction); (2) **multiplexed warm pools** — they keep a small pool of warm containers for popular models, *shared across customers*; (3) **shared model weights** — for popular open-source models, weights live on attached fast storage so loading is amortized across all tenants.

## The four 2026 platforms

| Platform | Best at | Cold start | Pricing model | Notes |
|---|---|---|---|---|
| **Modal** | Python-native ergonomics; ML/research workflows | ~12s (snapshot); sub-second warm | per-second GPU + per-second CPU + per-GB egress | Strongest dev-experience. $87M Series B 2025. Backend snapshotting tech is genuinely impressive |
| **RunPod Serverless** | Cheapest pay-per-second; many GPU types | 48% under 200ms (claimed) | per-second GPU only; no platform fee | Closest to "raw compute" — least abstraction overhead. Best for cost-optimizing |
| **Replicate** | Hosted open-source models out of the box | ~0 for shared models; 30-60s for custom | per-second + per-prediction | Acquired by Cloudflare Nov 2025. Best for shipping a community model fast |
| **Beam Cloud** | Mid-range; Tigris storage for fast cold starts | 2-3s typical; 50ms warm | per-second GPU + storage | Newer entrant. Good for medium-team production. |

What's *not* in the table because they died: **Banana.dev** (shut down late 2023/early 2024).

## When to use what — the decision tree

```
                    ┌─ steady-state QPS > 50? ──── yes ──► owned K8s (Topic 10)
                    │                                      ($/Mtok wins at scale)
                    │
   Pick substrate ──┤── spiky/low/zero traffic ─── yes ──► serverless (this topic)
                    │                                      (scale-to-zero saves you)
                    │
                    └── shipping a community model? ────► Replicate (zero ops)
                        first time deploying any LLM? ──► Modal (best DX)
                        cost-paranoid, willing to glue? ─► RunPod Serverless
                        multi-region failover required? ─► Beam or owned multi-cloud
```

The senior-eng lesson: **the same workload can be cheapest on three different substrates depending on QPS shape.** A chatbot with a 10× peak-to-trough ratio is *much* cheaper on serverless (you pay 1/10 of an always-on warm pool). A high-traffic OpenAI-compatible API serving 200+ QPS steady-state is dramatically cheaper on owned infra (no PaaS markup). The 2026 production answer is often *both*: owned infra for the hot path, serverless for the long tail of fine-tunes/experimental models.

## What the PaaS owns that you don't have to

Topic 11 walked you through the seven cold-start phases (process start → torch import → model load → graph capture → server ready). On serverless:

| Phase | Owned K8s | Serverless |
|---|---|---|
| Image pull | you DaemonSet pre-pull | PaaS handles |
| Process start | seconds | snapshotted |
| Torch import | ~1.3s | snapshotted |
| Model load | 30-60s (or use Run:ai streamer) | snapshotted (weights on attached fast storage) |
| Graph capture | ~8s | snapshotted |
| Server ready | total ~70s naive | total ~12s on Modal |
| Warm pool overhead | you pay for idle replicas | none |

You're paying ~30-50% PaaS markup for: the snapshotting tech (real engineering), the multi-tenant fleet (real GPU efficiency), and the zero ops burden (real eng-time saving). The math is favorable below a threshold and unfavorable above.

## Quickstart

```bash
# 1. Modal — install + deploy
pip install modal
modal token new
modal deploy modal_app.py
# Hit the URL Modal prints; first request: ~12s cold start. Second: ~200ms.

# 2. RunPod — handler-style
pip install runpod
# edit runpod_handler.py with your model
runpod project deploy

# 3. Replicate — push a hosted model
pip install replicate
# create a cog.yaml, run `cog push r8.im/yourname/yourmodel`

# 4. Comparison run
python compare_substrates.md  # actually a markdown writeup with measured numbers
```

## Expected output (illustrative — your numbers will vary)

```
Workload: Llama-3-7B FP16, 50 RPS for 60s then 0 RPS for 600s, repeat 3×

substrate              cold_start  warm_p50  warm_p99   $/Mtok      eng_hours_setup
─────────────────────────────────────────────────────────────────────────────────
K8s + vLLM (Topic 10)    73s        110ms     290ms     $0.83          ~16h
Modal                    11s        130ms     340ms     $1.42           ~1h
RunPod Serverless        4s         140ms     380ms     $1.05           ~3h
Replicate (shared 7B)    0s         180ms     440ms     $2.10           ~30min
Beam                     3s         145ms     360ms     $1.25           ~2h
```

**Read the table.** K8s wins on cost-at-scale and warm-latency. Serverless wins on cold-start, ops-burden, and time-to-first-deploy. Replicate wins on zero-effort if your model is a community model.

## Try

1. **Sweep peak/trough ratio.** Run the same workload with peak-to-trough ratios 1× (steady), 5×, 20×, 100×. Compute $/Mtok on K8s with `minReplicas` set to handle the trough vs serverless with scale-to-zero. Find the crossover ratio where serverless starts winning. On most realistic workloads it's between 5× and 15×.
2. **Cold-start parity test.** Spend a day engineering K8s cold-start down (Run:ai streamer + DaemonSet pre-pull + tensorizer + warmup). Compare to Modal out-of-the-box. Quantify how much engineering you'd have to do to beat Modal's snapshotting on your own infra. (Most teams stop short.)
3. **Migration thought-exercise.** Take Project 3's `mini-platform` and write a one-pager: *"At what QPS would we migrate from Modal to our own infra?"* Use your cost matrix from Topic 13. This is the doc real platform teams write at the Series B-to-C transition.
4. **Read [Modal's "How we achieved truly serverless GPUs" post](https://modal.com/blog/truly-serverless-gpus)** — the snapshotting tech writeup. Best public engineering blog on serverless GPU mechanics.

## Where this goes

- Topic 10 (`autoscaling-keda`) — the K8s answer to the same problem. Read Topic 10 → Topic 11 → 11b in sequence to see the design space.
- Topic 11 (`cold-start-and-warmup`) — Topic 11 owns the problem; 11b *delegates* it. Same seven phases, different owner.
- Topic 13 (`cost-economics`) + [CAPACITY-PLANNING.md](../13-cost-economics/CAPACITY-PLANNING.md) — the cost math needs an extra column for "serverless markup." For spiky workloads, the right column wins.
- Level 10 (System Design Capstone) — *"design an LLM inference service for a startup with 10 model variants and unknown traffic"* is a canonical question. The right answer in 2026 starts with "Modal or RunPod" and migrates to owned infra when economics force it.

## References

- Modal — *How we achieved truly serverless GPUs* — https://modal.com/blog/truly-serverless-gpus
- Modal docs — https://modal.com/docs
- RunPod Serverless docs — https://docs.runpod.io/serverless/overview
- Replicate docs — https://replicate.com/docs
- Beam Cloud docs — https://docs.beam.cloud/
- BuildMVPfast — *Serverless GPU comparison 2026* — https://www.buildmvpfast.com/blog/serverless-gpu-ai-inference-platform-comparison-2026
- Spheron — *10 Best Modal Alternatives in 2026* — https://www.spheron.network/blog/modal-alternatives/

## What's notable in the 2026 market

- **Modal Series B Sept 2025** ($87M @ $1.1B valuation) — serverless GPU is a real category now, not a side bet
- **Cloudflare acquired Replicate Nov 2025** — bundling serverless GPU into the CDN/edge story
- **Modular acquired BentoML Feb 2026** — the OSS serving frameworks are consolidating into PaaS plays
- **Mistral acquired Koyeb** — even model labs are picking up serving infra

Watch this market — it's where the platform layer for LLMs is being defined.

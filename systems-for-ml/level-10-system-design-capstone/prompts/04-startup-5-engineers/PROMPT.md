# Prompt 04 — Inference Platform for a 5-Engineer Startup

You're employee #6 at a Series A startup. The product is a B2B SaaS that helps support teams draft replies — it uses one fine-tuned 13B model + the company plans to ship 2–3 new fine-tunes per month for industry-specific variants.

Current traffic: ~40 QPS at peak (US business hours), ~2 QPS at night. Projected to 5× in the next 6 months if a key customer signs. You and four other engineers own everything from product to infra. CTO wants p95 < 1.5s end-to-end and tightest possible $/Mtok within engineering-time constraints.

**Design the inference platform.** Whiteboard the architecture, defend your substrate choice, do the cost math, and tell us what you'd change at 10× scale.

45-minute interview format. Clarifying questions are encouraged.

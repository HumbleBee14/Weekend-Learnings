# Prompt 02 — Chatbot at 10K QPS, p99 < 2s

You're the lead inference architect at a consumer chatbot company (think Character.AI scale, or Claude.ai's free tier). Traffic:

- **Peak: 10,000 QPS**, sustained from 5pm–11pm in each timezone
- **Trough: 500 QPS** at 4am
- Average input: 600 tokens; average output: 200 tokens
- **p99 end-to-end latency:** < 2 seconds (p50 < 400ms ideal)
- Model: Llama-3-70B fine-tuned for chat
- Cost target: aggressive — every $0.01 / Mtok saved is $100K / year at this scale
- 100M MAUs across US + EU

Design the inference platform. 45-minute interview format.

# Prompt 01 — Design Bedrock / Vertex AI Inference Equivalent

You've joined a big-3 cloud provider as the lead architect for their new managed inference offering. The goal: a customer can call your API with `{"model": "llama-3-70b", "input": "..."}` and get a streamed response, just like AWS Bedrock or Google Vertex AI Inference.

Requirements:
- **30+ models** in the catalog (open-source LLMs + a few in-house fine-tunes)
- **Multi-tenant** — thousands of customers, per-customer billing, per-customer rate limits
- **5-region** deployment (US-East, US-West, EU, APAC, India)
- **Latency SLOs:** p95 < 800ms TTFT for 7B-class, < 2s for 70B-class
- **99.95% availability** SLA
- **OpenAI-compatible** API surface (the contract your customers will speak)
- Cost target: competitive with Together AI and Fireworks on $/Mtok

Design the platform. 45-minute interview format.

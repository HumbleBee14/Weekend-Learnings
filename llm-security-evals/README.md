# LLM Security & Evals

Learning LLM security, red-teaming, evaluation frameworks, and AI governance — from attack fundamentals to production-grade defense systems.

## Why This Matters (2026)

This is **the** highest-demand skill gap in AI engineering right now:

| Signal | Data point |
|--------|-----------|
| Attack surface | 73% of production AI deployments have prompt injection vulnerabilities |
| Incidents | 520 tool-misuse incidents in 2026 alone (340% increase from 2024) |
| Salary premium | AI security engineers earn 43% more than traditional security roles |
| Supply gap | 1.3M AI job openings in US, fewer than 645K qualified candidates |
| Regulation | EU AI Act enforcement begins August 2026 — fines up to 7% of global revenue |
| Market signal | Lakera acquired by Check Point, Protect AI by Palo Alto Networks (2025) |

## What You'll Build

By the end of this course, you'll have built:

1. **A vulnerable LLM app** — intentionally exploitable, then hardened
2. **A red-teaming pipeline** — automated attack suite using Garak + PyRIT
3. **A guardrails system** — input/output scanning with NeMo Guardrails + LLM Guard
4. **An eval harness** — CI/CD-integrated quality gates using DeepEval + Promptfoo
5. **A secure RAG pipeline** — with poisoning defenses and retrieval-native auth
6. **A secure agent system** — with tool sandboxing and privilege boundaries
7. **A governance dashboard** — compliance checks mapped to OWASP + EU AI Act

## The Industry Toolbox

These are the tools companies actually use — you'll learn all of them:

### Red-Teaming & Offensive

| Tool | Maintainer | What it does |
|------|-----------|-------------|
| **Garak** | NVIDIA | LLM vulnerability scanner — "nmap for LLMs" |
| **PyRIT** | Microsoft | Multi-turn attack orchestration |
| **DeepTeam** | Confident AI | Red-teaming framework for RAG, chatbots, agents |
| **Promptfoo** | Open-source | Eval + red-team + security scanning, CI/CD native |

### Defense & Guardrails

| Tool | Maintainer | What it does |
|------|-----------|-------------|
| **NeMo Guardrails** | NVIDIA | Colang DSL for dialog flow control and safety policies |
| **LLM Guard** | Open-source | 15 input scanners + 20 output scanners, self-hosted |
| **Guardrails AI** | Open-source | Composable validation pipelines |
| **Llama Guard 3** | Meta | LLM-based safety classifier |
| **LlamaFirewall** | Meta | Orchestration layer for multi-guard systems |

### Evals & Quality

| Tool | Maintainer | What it does |
|------|-----------|-------------|
| **DeepEval** | Confident AI | 60+ metrics, Pytest-like, self-explaining scores |
| **Promptfoo** | Open-source | YAML-based eval + red-team, zero cloud deps |
| **RAGAS** | Open-source | RAG-specific eval metrics |
| **Inspect AI** | UK AISI | 100+ pre-built government-grade evaluations |
| **Braintrust** | Commercial | Full lifecycle: eval + observability + CI/CD gates |
| **LangSmith** | LangChain | Tracing + eval + production monitoring |

### Observability

| Tool | Maintainer | What it does |
|------|-----------|-------------|
| **Langfuse** | Open-source | Tracing + prompt management + eval |
| **Helicone** | Open-source | AI gateway, lightweight observability |
| **Arize Phoenix** | Arize AI | Notebook-first observability + drift detection |

## Setup

```bash
cd llm-security-evals
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash

# Core dependencies
pip install openai anthropic transformers torch

# Red-teaming tools
pip install garak pyrit deepteam promptfoo

# Guardrails
pip install nemoguardrails llm-guard guardrails-ai

# Evals
pip install deepeval ragas

# Observability
pip install langfuse

# Freeze
pip freeze > requirements.txt
```

> **Note**: Some tools (Garak, PyRIT) need specific Python versions. Each notebook will note its own requirements at the top.

## Structure

```
llm-security-evals/
│
├── level-1-llm-internals/
│   ├── 01-how-llms-work.ipynb                ← tokenization, attention, decoding — the attack surface
│   ├── 02-system-prompts-and-roles.ipynb      ← how system/user/assistant roles work internally
│   ├── 03-temperature-and-sampling.ipynb      ← how generation params affect exploitability
│   └── 04-tool-use-and-function-calling.ipynb ← how tools work, why they're exploitable
│
├── level-2-prompt-injection/
│   ├── 01-direct-prompt-injection.ipynb       ← override system prompt, role confusion, delimiter attacks
│   ├── 02-indirect-prompt-injection.ipynb     ← malicious content in retrieved docs, emails, websites
│   ├── 03-jailbreaking-techniques.ipynb       ← DAN, multi-turn escalation, persona attacks, encoding tricks
│   ├── 04-data-exfiltration.ipynb             ← extracting system prompts, PII, training data
│   └── 05-build-vulnerable-app.ipynb          ← build an intentionally vulnerable chatbot, then exploit it
│
├── level-3-red-teaming/
│   ├── 01-garak-scanner.ipynb                 ← install, configure, run scans, interpret results
│   ├── 02-pyrit-orchestration.ipynb           ← multi-turn attacks, attack strategies, scoring
│   ├── 03-deepteam-framework.ipynb            ← red-team RAG systems, chatbots, agents
│   ├── 04-promptfoo-red-team.ipynb            ← YAML-based red-team definitions, CI integration
│   ├── 05-manual-red-teaming.ipynb            ← structured manual testing methodology (OWASP + MITRE ATLAS)
│   └── 06-build-red-team-pipeline.ipynb       ← combine tools into automated red-team CI pipeline
│
├── level-4-guardrails-defense/
│   ├── 01-input-output-scanning.ipynb         ← LLM Guard scanners — PII, toxicity, prompt injection detection
│   ├── 02-nemo-guardrails.ipynb               ← Colang DSL, dialog rails, topic control, fact-checking
│   ├── 03-guardrails-ai-validators.ipynb      ← composable validators, Guardrails Hub, custom validators
│   ├── 04-llama-guard-classifier.ipynb        ← safety classification with Llama Guard 3
│   ├── 05-defense-in-depth.ipynb              ← layering: structural validation + content rails + security scan
│   └── 06-harden-vulnerable-app.ipynb         ← take the Level 2 vulnerable app and defend it
│
├── level-5-evals-framework/
│   ├── 01-eval-fundamentals.ipynb             ← what to eval: correctness, hallucination, safety, bias
│   ├── 02-deepeval-metrics.ipynb              ← 60+ metrics, Pytest integration, self-explaining scores
│   ├── 03-llm-as-judge.ipynb                  ← pairwise comparison, pointwise scoring, judge reliability
│   ├── 04-promptfoo-evals.ipynb               ← YAML test definitions, assertions, scoring
│   ├── 05-ragas-rag-evals.ipynb               ← context relevance, faithfulness, answer correctness
│   ├── 06-safety-evals.ipynb                  ← toxicity, bias, hallucination detection at scale
│   ├── 07-eval-driven-development.ipynb       ← TDD for AI: write evals first, iterate until they pass
│   └── 08-ci-cd-quality-gates.ipynb           ← GitHub Actions integration, block merges on regression
│
├── level-6-rag-security/
│   ├── 01-rag-attack-surface.ipynb            ← how RAG works, where attacks happen (ingest, retrieve, augment)
│   ├── 02-poisoned-rag-attack.ipynb           ← PoisonedRAG: 5 docs among millions → 90% attack success
│   ├── 03-indirect-injection-via-docs.ipynb   ← malicious instructions surviving vectorization
│   ├── 04-retrieval-native-auth.ipynb         ← access control at the retrieval layer, per-user filtering
│   ├── 05-ingestion-pipeline-defense.ipynb    ← source vetting, adversarial scans, PII redaction
│   └── 06-build-secure-rag.ipynb              ← end-to-end secure RAG pipeline with all defenses
│
├── level-7-agent-security/
│   ├── 01-owasp-agentic-top-10.ipynb          ← ASI01-ASI10: goal hijack, tool misuse, privilege abuse, rogue agents
│   ├── 02-tool-poisoning-attacks.ipynb        ← MCP tool poisoning, tool shadowing, supply chain attacks
│   ├── 03-sequential-tool-attack-chaining.ipynb ← STAC: individually harmless calls → collectively harmful
│   ├── 04-agent-sandboxing.ipynb              ← least-privilege, tool allowlists, permission boundaries
│   ├── 05-inter-agent-security.ipynb          ← secure communication, message authentication, trust boundaries
│   └── 06-build-secure-agent.ipynb            ← build a multi-tool agent with full security harness
│
├── level-8-governance-compliance/
│   ├── 01-owasp-llm-top-10.ipynb              ← LLM01-LLM10 deep dive with code examples for each
│   ├── 02-mitre-atlas.ipynb                   ← 16 tactics, 155 techniques — map attacks to defenses
│   ├── 03-eu-ai-act.ipynb                     ← risk classification, documentation requirements, compliance code
│   ├── 04-nist-ai-rmf.ipynb                   ← GOVERN, MAP, MEASURE, MANAGE — practical implementation
│   ├── 05-observability-monitoring.ipynb       ← Langfuse + Helicone setup, cost tracking, anomaly detection
│   └── 06-governance-dashboard.ipynb          ← build a compliance dashboard: OWASP checklist + audit logs
│
└── assets/
    └── images/
```

## Colab / GPU requirements by level

| Level | CPU ok? | Colab T4 | Notes |
|---|---|---|---|
| 1 | Yes | Not needed | Understanding internals, small local models |
| 2 | Yes | Not needed | Attack techniques work on API-based models |
| 3 | Mostly | Recommended for Garak | Some scanners benefit from local model |
| 4 | Yes (LLM Guard) | Required (Llama Guard) | Llama Guard needs GPU, others don't |
| 5 | Yes | Not needed | Evals run against API-based models |
| 6 | Yes | Recommended | Embedding + retrieval benefits from GPU |
| 7 | Yes | Not needed | Agent security is mostly API-based |
| 8 | Yes | Not needed | Governance is framework + monitoring code |

## Key References

### Standards
- [OWASP Top 10 for LLM Applications 2025](https://genai.owasp.org/resource/owasp-top-10-for-llm-applications-2025/)
- [OWASP Top 10 for Agentic Applications 2026](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/)
- [MITRE ATLAS](https://atlas.mitre.org/)
- [NIST AI Risk Management Framework](https://www.nist.gov/artificial-intelligence/risk-management-framework)
- [EU AI Act](https://artificialintelligenceact.eu/)

### Tools Documentation
- [Garak (NVIDIA)](https://github.com/NVIDIA/garak)
- [PyRIT (Microsoft)](https://github.com/Azure/PyRIT)
- [DeepTeam](https://github.com/confident-ai/deepteam)
- [DeepEval](https://github.com/confident-ai/deepeval)
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
- [LLM Guard](https://github.com/protectai/llm-guard)
- [Promptfoo](https://www.promptfoo.dev/)
- [RAGAS](https://github.com/explodinggradients/ragas)
- [Inspect AI (UK AISI)](https://inspect.aisi.org.uk/)
- [Langfuse](https://langfuse.com/)

### Research Papers
- "Prompt Injection Attacks in LLMs: A Comprehensive Review" (MDPI, 2025)
- "From Prompt Injections to Protocol Exploits" (ScienceDirect, 2025)
- "MCP Threat Modeling and Tool Poisoning" (arXiv:2603.22489)
- "PoisonedRAG" (USENIX Security 2025)
- "LLMs-as-Judges: A Comprehensive Survey" (arXiv:2412.05579)

### Certifications (when ready)
- **CAISP** — Certified AI Security Professional (Practical DevSecOps) — 15-20% salary premium
- **OSAI+** — Advanced AI Red Teaming (OffSec) — hands-on, lab-based
- **CompTIA SecAI+** — vendor-neutral AI security
- **SANS GOAA** — GIAC Offensive AI Analyst (new 2026)

## Resources

- [Awesome-LLM-Security (GitHub)](https://github.com/corca-ai/awesome-llm-security)
- [AI Red Teaming Guide (GitHub)](https://github.com/requie/AI-Red-Teaming-Guide)
- [The Vulnerable MCP Project](https://vulnerablemcp.info/)
- [OWASP GenAI Security Project](https://genai.owasp.org/)

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

By the end of this learning, we'll have built:

1. **A vulnerable LLM app** — intentionally exploitable, then hardened
2. **A red-teaming pipeline** — automated attack suite using Garak + PyRIT
3. **A guardrails system** — input/output scanning with NeMo Guardrails + LLM Guard
4. **An eval harness** — CI/CD-integrated quality gates using DeepEval + Promptfoo
5. **A secure RAG pipeline** — with poisoning defenses and retrieval-native auth
6. **A secure agent system** — with tool sandboxing and privilege boundaries
7. **A governance dashboard** — compliance checks mapped to OWASP + EU AI Act

## The Industry Toolbox

These are the tools we actually use — **25+ production tools** across the full stack:

### Red-Teaming & Offensive (5 tools)

| Tool | Maintainer | What it does | Notebook |
|------|-----------|-------------|----------|
| **Garak** | NVIDIA | LLM vulnerability scanner — "nmap for LLMs" | `level-3/01` |
| **PyRIT** | Microsoft | Multi-turn attack orchestration | `level-3/02` |
| **DeepTeam** | Confident AI | Red-teaming framework for RAG, chatbots, agents | `level-3/03` |
| **Promptfoo** | Open-source | Eval + red-team + security scanning, CI/CD native | `level-3/04` |
| **Inspect AI** | UK AISI | 100+ pre-built security evals, government-grade | `level-3/07` |

### Defense & Guardrails (7 tools)

| Tool | Maintainer | What it does | Notebook |
|------|-----------|-------------|----------|
| **LLM Guard** | Open-source | 15 input scanners + 20 output scanners, self-hosted | `level-4/01` |
| **NeMo Guardrails** | NVIDIA | Colang DSL for dialog flow control and safety policies | `level-4/02` |
| **Guardrails AI** | Open-source | Composable validation pipelines from Hub | `level-4/03` |
| **Llama Guard 3** | Meta | LLM-based safety classifier for content moderation | `level-4/04` |
| **Llama Prompt Guard 2** | Meta | Dedicated 86M-param prompt injection classifier | `level-4/05` |
| **LlamaFirewall** | Meta | Orchestration layer across multiple guard models | `level-4/06` |
| **Pydantic + Instructor** | Open-source | Structural output validation — first defense layer | `level-4/07` |

### Evals & Quality (6 tools)

| Tool | Maintainer | What it does | Notebook |
|------|-----------|-------------|----------|
| **DeepEval** | Confident AI | 60+ metrics, Pytest-like, self-explaining scores | `level-5/02` |
| **Promptfoo** | Open-source | YAML-based eval + red-team, zero cloud deps | `level-5/04` |
| **RAGAS** | Open-source | RAG-specific eval metrics (faithfulness, relevance) | `level-5/05` |
| **Inspect AI** | UK AISI | 100+ pre-built government-grade evaluations | `level-5/09` |
| **LangSmith** | LangChain | Multi-turn evals, online scoring, Insights Agent | `level-5/10` |
| **Braintrust** | Commercial | Full lifecycle: eval + observability + CI/CD gates | `level-5/11` |

### Observability (3 tools)

| Tool | Maintainer | What it does | Notebook |
|------|-----------|-------------|----------|
| **Langfuse** | Open-source | Tracing + prompt management + eval scoring | `level-8/obs-01` |
| **Helicone** | Open-source | AI gateway, routing, caching, cost tracking | `level-8/obs-02` |
| **Arize Phoenix** | Arize AI | Notebook-first observability + drift detection | `level-8/obs-02` |

### Standards & Frameworks (6 frameworks)

| Framework | Body | What it covers | Notebook |
|-----------|------|---------------|----------|
| **OWASP Top 10 for LLMs** | OWASP | LLM01-LLM10 vulnerability taxonomy (2025) | `level-8/01` |
| **OWASP Top 10 for Agentic Apps** | OWASP | ASI01-ASI10 agent-specific threats (2026) | `level-7/01` |
| **MITRE ATLAS** | MITRE | 16 tactics, 155 techniques, 35 mitigations | `level-8/02` |
| **EU AI Act** | European Union | Risk classification, enforcement Aug 2026 | `level-8/03` |
| **NIST AI RMF** | NIST | GOVERN, MAP, MEASURE, MANAGE | `level-8/04` |
| **ISO/IEC 42001** | ISO | AI Management Systems standard | `level-8/05` |

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
pip install pydantic instructor

# Evals
pip install deepeval ragas inspect-ai

# Observability
pip install langfuse helicone arize-phoenix

# Meta security stack (requires HuggingFace access)
pip install accelerate   # for Llama Guard / Prompt Guard

# Freeze
pip freeze > requirements.txt
```

> **Note**: Some tools (Garak, PyRIT) need specific Python versions. Each level has its own `requirements.txt`.

## Structure

Levels 1-2 use **notebooks** (exploratory, interactive attack testing).
Levels 3-8 use **production Python** (scripts, FastAPI apps, pytest suites, CI/CD configs).

```
llm-security-evals/
│
├── level-1-llm-internals/                        # NOTEBOOKS — exploratory
│   ├── 01-how-llms-work.ipynb                    ← tokenization, attention, decoding — the attack surface
│   ├── 02-system-prompts-and-roles.ipynb         ← how system/user/assistant roles work internally
│   ├── 03-temperature-and-sampling.ipynb         ← how generation params affect exploitability
│   └── 04-tool-use-and-function-calling.ipynb    ← how tools work, why they're exploitable
│
├── level-2-prompt-injection/                      # NOTEBOOKS + PYTHON PROJECT
│   ├── 01-direct-prompt-injection.ipynb          ← override system prompt, role confusion, delimiter attacks
│   ├── 02-indirect-prompt-injection.ipynb        ← malicious content in retrieved docs, emails, websites
│   ├── 03-jailbreaking-techniques.ipynb          ← DAN, multi-turn escalation, persona attacks, encoding tricks
│   ├── 04-data-exfiltration.ipynb                ← extracting system prompts, PII, training data
│   ├── vulnerable_app/                           ← FastAPI chatbot with intentional security holes
│   │   ├── main.py                               ← the vulnerable app — DVWA for LLMs
│   │   └── config.py                             ← intentionally insecure config
│   └── attacks/
│       └── run_attacks.py                        ← scripted attacks against the vulnerable app
│
├── level-3-red-teaming/                           # PYTHON SCRIPTS + CLI CONFIGS
│   ├── 01_garak_scanner.py                       ← NVIDIA's LLM vuln scanner
│   ├── 02_pyrit_orchestration.py                 ← Microsoft's multi-turn attack tool
│   ├── 03_deepteam_redteam.py                    ← red-team RAG, chatbots, agents
│   ├── 04_promptfoo_redteam.py                   ← YAML-based red-team + CI integration
│   ├── 05_manual_methodology.py                  ← structured manual testing (OWASP + MITRE ATLAS)
│   ├── 06_red_team_pipeline.py                   ← combine all tools into automated CI pipeline
│   ├── 07_inspect_ai_security.py                 ← UK AISI's 100+ security evals
│   └── configs/
│       ├── garak_config.yaml
│       ├── promptfoo_redteam.yaml
│       └── pyrit_config.yaml
│
├── level-4-guardrails-defense/                    # FASTAPI PROJECT — production guardrails
│   ├── app/
│   │   ├── main.py                               ← FastAPI app with defense middleware
│   │   ├── defense.py                            ← defense-in-depth: wires all layers together
│   │   └── middleware/
│   │       ├── 01_llm_guard.py                   ← input/output scanning (PII, toxicity, injection)
│   │       ├── 02_nemo_rails.py                  ← Colang DSL dialog flow control
│   │       ├── 03_guardrails_ai.py               ← composable validators from Hub
│   │       ├── 04_llama_guard.py                 ← Meta's safety classifier
│   │       ├── 05_prompt_guard.py                ← Meta's 86M-param injection classifier
│   │       ├── 06_llama_firewall.py              ← Meta's multi-guard orchestration
│   │       └── 07_structural.py                  ← Pydantic + Instructor output validation
│   ├── configs/nemo/
│   │   ├── config.yml                            ← NeMo Guardrails config
│   │   └── rails.co                              ← Colang rails definitions
│   └── tests/
│       └── test_guardrails.py                    ← verify each guardrail catches what it should
│
├── level-5-evals-framework/                       # PYTEST SUITE — evals as tests
│   ├── tests/
│   │   ├── conftest.py                           ← shared fixtures: LLM client, datasets
│   │   ├── test_01_fundamentals.py               ← eval taxonomy: correctness, hallucination, safety, bias
│   │   ├── test_02_deepeval.py                   ← 60+ metrics, Pytest-native
│   │   ├── test_03_llm_judge.py                  ← LLM-as-Judge: pairwise, pointwise, reliability
│   │   ├── test_04_promptfoo.py                  ← YAML eval definitions, assertions
│   │   ├── test_05_ragas.py                      ← RAG evals: faithfulness, relevance, correctness
│   │   └── test_06_safety.py                     ← toxicity, bias, hallucination at scale
│   ├── 07_eval_driven_dev.py                     ← TDD for AI: evals first, iterate until pass
│   ├── 08_ci_cd_gates.py                         ← GitHub Actions quality gate integration
│   ├── 09_inspect_ai_evals.py                    ← UK AISI's 100+ standardized evals
│   ├── 10_langsmith_evals.py                     ← LangChain's production eval platform
│   ├── 11_braintrust_evals.py                    ← CI/CD-native eval with GitHub Action gates
│   ├── configs/
│   │   ├── promptfoo_eval.yaml
│   │   └── deepeval_config.py
│   ├── .github/workflows/
│   │   └── eval-gate.yml                         ← actual GitHub Action: block merges on regression
│   └── pytest.ini
│
├── level-6-rag-security/                          # FULL RAG PROJECT — attack + defend
│   ├── app/
│   │   ├── main.py                               ← FastAPI secure RAG application
│   │   ├── rag/
│   │   │   ├── ingest.py                         ← ingestion with adversarial scanning + PII redaction
│   │   │   ├── retriever.py                      ← retrieval with per-user authorization
│   │   │   ├── augmenter.py                      ← augmentation with injection sanitization
│   │   │   └── scanner.py                        ← adversarial document scanner
│   │   └── auth/
│   │       └── retrieval_auth.py                 ← attribute-based access control for vector DB
│   ├── attacks/
│   │   ├── 01_attack_surface.py                  ← map every RAG attack point
│   │   ├── 02_poisoned_rag.py                    ← PoisonedRAG: 5 docs → 90% success
│   │   └── 03_indirect_injection.py              ← injection payloads that survive embedding
│   └── tests/
│       └── test_rag_security.py
│
├── level-7-agent-security/                        # FULL AGENT PROJECT — attack + defend
│   ├── agent/
│   │   ├── main.py                               ← secure multi-tool agent
│   │   ├── tools/
│   │   │   ├── sandbox.py                        ← capability-based tool sandboxing
│   │   │   └── registry.py                       ← tool allowlist + integrity verification
│   │   ├── security/
│   │   │   ├── permissions.py                    ← runtime permission boundaries
│   │   │   ├── inter_agent.py                    ← secure inter-agent communication
│   │   │   └── supply_chain.py                   ← dependency vetting + integrity checks
│   │   └── mcp/
│   │       └── security.py                       ← MCP-specific security controls
│   ├── attacks/
│   │   ├── 01_owasp_agentic.py                   ← OWASP ASI01-ASI10 exploit code
│   │   ├── 02_tool_poisoning.py                  ← MCP tool poisoning + shadowing
│   │   ├── 03_stac_attack.py                     ← sequential tool attack chaining
│   │   └── 04_mcp_exploits.py                    ← MCP CVE reproductions
│   └── tests/
│       └── test_agent_security.py
│
├── level-8-governance-compliance/                 # SCRIPTS + DASHBOARD PROJECT
│   ├── frameworks/
│   │   ├── 01_owasp_llm_top10.py                ← LLM01-LLM10 with exploit + defense code
│   │   ├── 02_mitre_atlas.py                     ← 16 tactics, 155 techniques mapping
│   │   ├── 03_eu_ai_act.py                       ← risk classification + compliance automation
│   │   ├── 04_nist_ai_rmf.py                     ← GOVERN, MAP, MEASURE, MANAGE
│   │   └── 05_iso_42001.py                       ← AI Management Systems + framework integration
│   ├── observability/
│   │   ├── 01_langfuse_setup.py                  ← tracing, cost tracking, anomaly detection
│   │   └── 02_helicone_phoenix.py                ← AI gateway + drift detection
│   └── dashboard/
│       ├── main.py                               ← governance dashboard (FastAPI + HTMX)
│       ├── checklist.py                          ← automated OWASP compliance checker
│       └── audit.py                              ← tamper-evident audit logging
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
- [Guardrails AI](https://github.com/guardrails-ai/guardrails)
- [Llama Guard / Prompt Guard / LlamaFirewall (Meta)](https://github.com/meta-llama/PurpleLlama)
- [Instructor](https://github.com/jxnl/instructor)
- [Promptfoo](https://www.promptfoo.dev/)
- [RAGAS](https://github.com/explodinggradients/ragas)
- [Inspect AI (UK AISI)](https://inspect.aisi.org.uk/)
- [LangSmith](https://www.langchain.com/langsmith)
- [Braintrust](https://www.braintrust.dev/)
- [Langfuse](https://langfuse.com/)
- [Helicone](https://github.com/Helicone/helicone)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)

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

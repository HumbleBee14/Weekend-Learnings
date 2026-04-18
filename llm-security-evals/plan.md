# LLM Security & Evals — Learning Plan

## The fast track (weekend warrior path)

If you only have limited weekends, do these in order. Each one builds on the previous.

| # | File | Time | What you get |
|---|------|------|-------------|
| 1 | `level-1/01-how-llms-work.ipynb` | 30 min | Understand the attack surface — tokenization, roles, tool calls |
| 2 | `level-2/01-direct-prompt-injection.ipynb` | 45 min | Your first attacks — override system prompts, extract secrets |
| 3 | `level-2/vulnerable_app/main.py` | 1 hr | Build a vulnerable FastAPI chatbot, break it yourself |
| 4 | `level-3/01_garak_scanner.py` | 45 min | Automated vulnerability scanning with NVIDIA's tool |
| 5 | `level-4/app/middleware/01_llm_guard.py` | 45 min | Defend with LLM Guard — scan inputs/outputs |
| 6 | `level-4/app/defense.py` | 1 hr | Wire up defense-in-depth — fix the app you broke in step 3 |
| 7 | `level-5/tests/test_02_deepeval.py` | 45 min | Write your first eval suite — Pytest for LLMs |
| 8 | `level-5/08_ci_cd_gates.py` | 45 min | Block bad deploys with automated quality gates |

**After this track**: you can attack LLMs, defend them, and eval them.

---

## The full curriculum

### Phase 1: Understand the Machine (Level 1)
*Before you can break it, understand how it works.*

| # | File | Focus | Java analogy |
|---|------|-------|-------------|
| 1.1 | `01-how-llms-work.ipynb` | Tokenization, attention, decoding | Understanding JVM internals before writing exploits |
| 1.2 | `02-system-prompts-and-roles.ipynb` | System/user/assistant roles, how they're actually processed | Like understanding servlet filters and request lifecycle |
| 1.3 | `03-temperature-and-sampling.ipynb` | How generation params affect predictability | Like understanding JVM GC tuning — same code, different behavior |
| 1.4 | `04-tool-use-and-function-calling.ipynb` | How function calling works under the hood | Like understanding Spring's proxy-based AOP — magic that's exploitable |

**Most of this overlaps with the PyTorch level-1 foundation.** Skim quickly, focus on the security lens.

---

### Phase 2: Learn to Attack (Levels 2-3)
*You can't defend what you can't break.*

#### Level 2 — Prompt Injection (the #1 vulnerability)
| # | File | Focus | Real-world parallel |
|---|------|-------|-------------------|
| 2.1 | `01-direct-prompt-injection.ipynb` | Override system prompts, role confusion | Like SQL injection — the input becomes the command |
| 2.2 | `02-indirect-prompt-injection.ipynb` | Poison via external content (docs, emails) | Like stored XSS — the payload lives in the data |
| 2.3 | `03-jailbreaking-techniques.ipynb` | DAN, multi-turn escalation, encoding tricks | Like WAF bypass techniques — evade the filter |
| 2.4 | `04-data-exfiltration.ipynb` | Extract system prompts, PII, training data | Like SSRF — make the system leak its own secrets |
| 2.5 | `vulnerable_app/main.py` | Build a FastAPI chatbot with real vulns, exploit them | Like DVWA but for LLMs |
| 2.6 | `attacks/run_attacks.py` | Scripted attacks against the vulnerable app | Automated exploit verification |

#### Level 3 — Red-Teaming (automated + manual)
| # | File | Focus | Real-world parallel |
|---|------|-------|-------------------|
| 3.1 | `01_garak_scanner.py` | NVIDIA's LLM vuln scanner | Like running nmap + nikto on a web app |
| 3.2 | `02_pyrit_orchestration.py` | Microsoft's multi-turn attack tool | Like Burp Suite's intruder — sophisticated, scriptable |
| 3.3 | `03_deepteam_redteam.py` | Red-team RAG, chatbots, agents | Like OWASP ZAP but for AI systems |
| 3.4 | `04_promptfoo_redteam.py` | YAML-based red-team + CI integration | Like Semgrep — shift-left security scanning |
| 3.5 | `05_manual_methodology.py` | Structured methodology using OWASP + MITRE ATLAS | Like manual pentesting with OWASP Testing Guide |
| 3.6 | `06_red_team_pipeline.py` | Combine all tools into CI pipeline | Your capstone: automated security testing on every PR |
| 3.7 | `07_inspect_ai_security.py` | UK AISI's 100+ pre-built security evals | Like NIST's compliance test suites — government-grade |

All Level 3 scripts target **Level 2's vulnerable app** as their default attack surface.

---

### Phase 3: Learn to Defend (Level 4)
*Now build the walls. This is a FastAPI project — each middleware module is a guardrail.*

| # | File | Focus | Real-world parallel |
|---|------|-------|-------------------|
| 4.1 | `app/middleware/01_llm_guard.py` | LLM Guard — PII, toxicity, injection detection | Like a WAF (Web Application Firewall) for LLMs |
| 4.2 | `app/middleware/02_nemo_rails.py` | Colang DSL, dialog rails, topic control | Like Spring Security's filter chain — declarative rules |
| 4.3 | `app/middleware/03_guardrails_ai.py` | Composable validators from Guardrails Hub | Like Bean Validation (@Valid, custom validators) |
| 4.4 | `app/middleware/04_llama_guard.py` | Meta's safety classifier model | Like a dedicated security microservice |
| 4.5 | `app/middleware/05_prompt_guard.py` | Meta's 86M-param injection classifier | Like a dedicated WAF rule engine — purpose-built for one job |
| 4.6 | `app/middleware/06_llama_firewall.py` | Meta's multi-guard orchestration layer | Like an API Gateway chaining security filters |
| 4.7 | `app/middleware/07_structural.py` | Structural output validation (Pydantic + Instructor) | Like Jackson @JsonProperty + Bean Validation — schema-first |
| 4.8 | `app/defense.py` | Wire all layers into defense-in-depth | Like the security onion: WAF + auth + validation + logging |
| 4.9 | `app/main.py` | The hardened FastAPI app (secures Level 2's vulnerable app) | Full circle: break it, then fix it |

---

### Phase 4: Learn to Evaluate (Level 5)
*Evals are the new unit tests. If you can't measure it, you can't ship it.*

| # | File | Focus | Java analogy |
|---|------|-------|-------------|
| 5.1 | `tests/test_01_fundamentals.py` | What to eval: correctness, hallucination, safety, bias | Like understanding test categories: unit, integration, E2E |
| 5.2 | `tests/test_02_deepeval.py` | 60+ metrics, Pytest integration | Like JUnit + AssertJ — rich assertion library |
| 5.3 | `tests/test_03_llm_judge.py` | Use one LLM to evaluate another | Like property-based testing — the oracle IS an LLM |
| 5.4 | `tests/test_04_promptfoo.py` | YAML test definitions, assertions | Like TestContainers — declarative, reproducible |
| 5.5 | `tests/test_05_ragas.py` | RAG-specific: faithfulness, relevance, correctness | Like testing a search engine's result quality |
| 5.6 | `tests/test_06_safety.py` | Toxicity, bias, hallucination at scale | Like OWASP dependency check but for model outputs |
| 5.7 | `07_eval_driven_dev.py` | Write evals first, iterate until pass | Literally TDD for AI — @Test but semantic |
| 5.8 | `08_ci_cd_gates.py` | GitHub Actions, block merges on regression | Like SonarQube quality gates — no deploy if quality drops |
| 5.9 | `09_inspect_ai_evals.py` | UK AISI's 100+ standardized eval suite | Like JMH benchmarks — standardized, reproducible |
| 5.10 | `10_langsmith_evals.py` | LangChain's production eval platform | Like Datadog APM — production monitoring + eval |
| 5.11 | `11_braintrust_evals.py` | CI/CD-native eval with GitHub Action gates | Like SonarCloud — quality gate as a service |

Run tests with: `pytest tests/ -v`

---

### Phase 5: Secure Real Systems (Levels 6-7)
*Production is where it gets real. Each level is a full Python project.*

#### Level 6 — RAG Security
| # | File | Focus |
|---|------|-------|
| 6.1 | `attacks/01_attack_surface.py` | Map every attack point: ingest, retrieve, augment, generate |
| 6.2 | `attacks/02_poisoned_rag.py` | 5 poisoned docs among millions → 90% attack success |
| 6.3 | `attacks/03_indirect_injection.py` | Malicious instructions that survive embedding |
| 6.4 | `app/auth/retrieval_auth.py` | Per-user access control at the vector DB level |
| 6.5 | `app/rag/ingest.py` | Ingestion defense: source vetting, adversarial scans, PII redaction |
| 6.6 | `app/rag/retriever.py` | Retrieval with native authorization |
| 6.7 | `app/rag/augmenter.py` | Augmentation with injection sanitization |
| 6.8 | `app/rag/scanner.py` | Adversarial document scanner |
| 6.9 | `app/main.py` | **Capstone**: end-to-end secure RAG FastAPI app |

#### Level 7 — Agent Security (the 2026 frontier)
| # | File | Focus |
|---|------|-------|
| 7.1 | `attacks/01_owasp_agentic.py` | ASI01-ASI10: the new threat model for agents |
| 7.2 | `attacks/02_tool_poisoning.py` | MCP tool poisoning, shadowing, metadata injection |
| 7.3 | `attacks/03_stac_attack.py` | Individually harmless calls → collectively harmful |
| 7.4 | `attacks/04_mcp_exploits.py` | MCP CVE reproductions from The Vulnerable MCP Project |
| 7.5 | `agent/tools/sandbox.py` | Least-privilege, tool allowlists, capability-based security |
| 7.6 | `agent/tools/registry.py` | Tool allowlist + integrity verification |
| 7.7 | `agent/security/permissions.py` | Runtime permission boundaries |
| 7.8 | `agent/security/inter_agent.py` | Secure comms, message auth, trust boundaries |
| 7.9 | `agent/security/supply_chain.py` | OpenClaw crisis, LiteLLM compromise, dependency vetting |
| 7.10 | `agent/mcp/security.py` | MCP-specific security controls |
| 7.11 | `agent/main.py` | **Capstone**: secure multi-tool agent with full harness |

---

### Phase 6: Governance & Production (Level 8)
*Ship it to prod, keep it compliant.*

| # | File | Focus |
|---|------|-------|
| 8.1 | `frameworks/01_owasp_llm_top10.py` | LLM01-LLM10 deep dive with exploit + defense code |
| 8.2 | `frameworks/02_mitre_atlas.py` | Map attacks to the 16 tactics, 155 techniques framework |
| 8.3 | `frameworks/03_eu_ai_act.py` | Risk classification, documentation, compliance checks |
| 8.4 | `frameworks/04_nist_ai_rmf.py` | GOVERN, MAP, MEASURE, MANAGE — practical code |
| 8.5 | `frameworks/05_iso_42001.py` | ISO/IEC 42001 AI Management Systems + framework integration |
| 8.6 | `observability/01_langfuse_setup.py` | Langfuse: tracing, cost tracking, anomaly detection |
| 8.7 | `observability/02_helicone_phoenix.py` | AI gateway observability + drift detection |
| 8.8 | `dashboard/main.py` | **Final capstone**: governance dashboard (FastAPI + HTMX) |
| 8.9 | `dashboard/checklist.py` | Automated OWASP compliance checker |
| 8.10 | `dashboard/audit.py` | Tamper-evident audit logging (EU AI Act requirement) |

---

## Suggested weekend schedule

| Weekend | What to do | Hours |
|---------|-----------|-------|
| 1 | Level 1 (all) — skim fast, you know most of this from PyTorch | 2 hrs |
| 2 | Level 2 (2.1-2.4) — learn to attack in notebooks | 3 hrs |
| 3 | Level 2 (2.5-2.6) + Level 3 (3.1-3.2) — build vulnerable app + automated scanning | 3 hrs |
| 4 | Level 3 (3.3-3.7) — full red-team pipeline + Inspect AI | 4 hrs |
| 5 | Level 4 (4.1-4.4) — core defense middleware modules | 4 hrs |
| 6 | Level 4 (4.5-4.9) — Meta stack + Pydantic + defense-in-depth | 4 hrs |
| 7 | Level 5 (5.1-5.6) — eval tests: DeepEval, LLM-as-Judge, RAGAS, safety | 4 hrs |
| 8 | Level 5 (5.7-5.11) — EDD, CI/CD gates, production platforms | 4 hrs |
| 9 | Level 6 (6.1-6.9) — RAG attacks + secure RAG capstone | 5 hrs |
| 10 | Level 7 (7.1-7.4) — agent threat model + attacks | 4 hrs |
| 11 | Level 7 (7.5-7.11) — agent defense + MCP security + capstone | 4 hrs |
| 12 | Level 8 (all) — governance, compliance, observability + final capstone | 5 hrs |

**12 weekends, ~46 hours total.** After this you can:
- Red-team any LLM system
- Defend it with production-grade guardrails
- Eval it with automated quality gates in CI/CD
- Secure RAG pipelines and multi-tool agents
- Map everything to OWASP, MITRE ATLAS, EU AI Act, NIST AI RMF, ISO 42001

## What to do next

1. **Contribute**: Submit probes to Garak, validators to Guardrails Hub, evals to Inspect AI
2. **Red-team for real**: Apply to bug bounty programs (OpenAI, Anthropic, Google run them)
3. **Go deeper**: Pick a domain (finance, healthcare, infra) and apply everything to a real system

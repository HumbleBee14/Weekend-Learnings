# Threat Model — `local-agent`

> Fill this in as the privacy section of `reports/local.md`. Be specific.
> Vague claims here are how "private" stacks become not-private in production.

## 1. Models in use

| Role | Model | Where it runs | Framework | Posture |
|------|-------|---------------|-----------|---------|
| autocomplete | _e.g._ Qwen2.5-Coder 7B 4-bit | local | MLX | on-device |
| chat | _e.g._ Qwen3-Coder 32B 4-bit | local | Ollama-MLX | on-device |
| escalation | _e.g._ Claude / GPT-4 | vendor cloud | OpenAI client | cloud LLM |
| system | _e.g._ Foundation Models 3B | local + PCC | FoundationModels | on-device with PCC fallback |

## 2. Tools the agent can call

For each tool: does the call leave the device? If yes, what data goes where?

| Tool | Network? | Endpoint | Data sent | Notes |
|------|----------|----------|-----------|-------|
| file_read | no | n/a | n/a | local FS |
| file_edit | no | n/a | n/a | local FS |
| shell_exec | no* | n/a | n/a | _*shells out can themselves call network_ |
| web_search | yes | _e.g._ duckduckgo.com | the search query | not "private" — query leaves device |
| send_email | yes | smtp host | the message | obvious; document anyway |

## 3. Logs and on-disk artifacts

| Component | What it logs | Default sensitivity | Mitigation |
|-----------|--------------|---------------------|------------|
| Ollama daemon | requests, model, timing | metadata; no prompt by default | confirm `OLLAMA_DEBUG=0` |
| llama-server | full prompts and outputs at -lv 2 | high | run at default verbosity, redirect to /dev/null |
| agent loop | tool calls, tool results | high (results carry doc content) | redact before persisting; opt-in debug mode |
| KV cache snapshots | last conversation | medium | encrypt at rest or do not persist |
| Fine-tuning data | full training set | high | encrypt at rest; never sync to iCloud |

## 4. Process permissions

| Permission | Granted? | Why | Threat surface widened |
|------------|----------|-----|------------------------|
| Full Disk Access | _y/n_ | _agent reads project files_ | yes — any malicious code in process now reads everything |
| Accessibility | _y/n_ | _autocomplete in IDE_ | reads keystrokes |
| Screen Recording | _y/n_ | _vision agent_ | reads everything on screen |
| Microphone | _y/n_ | _voice mode_ | always-on capture surface |

## 5. Posture per request type

State the routing rule and the user's visibility into it:

- **Tier 1 (default):** _e.g._ on-device 7B handles all autocomplete and short-form chat.
- **Tier 2 (verifier-escalated):** _e.g._ if local model returns low-confidence or refusal, prompt the user before falling back to PCC or cloud.
- **Tier 3 (explicit):** _e.g._ user types `/cloud` to force a cloud call; banner is shown.
- **UI:** _e.g._ tray icon shows current tier in real time.

## 6. Honest limits

What this stack does **not** protect against (state explicitly):

- Compromised device / root malware: out of scope.
- Same-device snooping via Accessibility/Screen Recording APIs granted to other apps.
- Physical seizure of an unlocked machine.
- Network leakage from explicitly-network-using tools (web search, email).
- _your-list_

## 7. Verification

How would a third party (security reviewer, auditor, paranoid user) verify the
above? List concrete checks:

- _e.g._ run `lsof -i` while the agent is idle; confirm only expected sockets.
- _e.g._ run a tcpdump capture during a chat session; confirm no traffic to vendor APIs unless cloud tier is engaged.
- _e.g._ inspect Ollama and llama-server logs after a session; confirm no prompt content.

## 8. Comparison to PCC and to cloud

A short paragraph: of the three postures (on-device / PCC / cloud), which does
each model use, and where does that put the overall product? Reference Topic 16
CONCEPTS.md for definitions.

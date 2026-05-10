# 16 — Privacy and Private Cloud Compute

## Three postures, not two

The pop-culture framing of "on-device good, cloud bad" loses the interesting middle. In 2026 there are three meaningfully different privacy postures any local agent must reason about:

| Posture | Where compute happens | Trust root | Verifiable from outside? |
|---------|------------------------|------------|--------------------------|
| **On-device** | Your CPU/GPU/ANE | Your device's Secure Enclave | No (device is opaque to others) |
| **Private Cloud Compute (PCC)** | Apple Silicon servers in Apple DCs | Hardware root of trust + binary transparency | Yes (anyone can audit the published images) |
| **Cloud LLM** | Vendor servers (OpenAI, Anthropic, Google, ...) | Vendor's word + their subprocessors | Mostly no — SOC 2 reports, contractual terms |

These are not on a linear scale. They are different threat models with different failure modes.

## What on-device actually buys

- Data never leaves the device. No vendor server sees the prompt, no logs exist that can be subpoenaed from a third party, no risk of training-data leakage from your queries.
- Offline operation. The model still runs on a flight or in a dead zone.
- No per-token cost. Latency floor is yours to control.
- Zero correlation across users. Apple's PCC and any cloud LLM see traffic from millions; on-device queries are inherently isolated.

## What on-device does **not** buy

- **Protection from a compromised device.** Root malware reads RAM. The Secure Enclave protects keys, not the entire userland process running the model.
- **Protection from same-device snooping.** Accessibility APIs, screenshot tools, screen recording, clipboard observers — anything granted those permissions can read prompts and outputs.
- **Protection from physical seizure.** A powered-on unlocked Mac is a powered-on unlocked Mac.
- **Protection from network-leaking integrations.** Your local agent is only on-device if **every tool it calls** is also local. A web-search tool, a calendar API, a "send to Slack" tool — those are network requests with their own privacy surfaces.
- **Auditable correctness.** You cannot prove to a third party that your local stack actually ran on-device. Your compliance team has to take your word for it.

That last point is where PCC enters.

## Apple Private Cloud Compute — what it actually is

PCC is Apple's design for running LLM inference on cloud Apple Silicon while preserving most of the on-device privacy guarantees through cryptographic attestation rather than policy.

The architecture, in one diagram:

```
  Device                         Apple DC: PCC node
  +----------------+             +----------------------------------+
  | Foundation     |   TLS +     |  Secure Boot + measured launch    |
  | Models +       |   client    |  Sealed enclave; no SSH; no       |
  | request        |  attest    |  persistent storage; per-request  |
  | + ephemeral    | -------->   |  ephemeral memory only            |
  | symmetric key  |             |                                   |
  +----------------+             |  Inference runs against published |
                                  |  signed image; key destroyed at   |
                                  |  end of request.                  |
                                  +----------------------------------+
                                            ^
                                            |
                       +---------------------+
                       | Public Binary       |
                       | Transparency Log    |
                       | (researchers diff   |
                       | every shipped image)|
                       +---------------------+
```

The five claims Apple makes about PCC:

1. **Stateless computation on personal data.** Request data is held only in volatile memory and only while the request is being processed. Nothing persists.
2. **Enforceable guarantees, not policy.** The constraints are enforced by the hardware (Secure Boot, sealed memory, no admin interfaces) — not by policy or contract.
3. **No privileged runtime access.** Apple SREs cannot SSH in. There is no "fix it in prod" path.
4. **Non-targetability.** A device cannot be served by a specific PCC node identified in advance — node selection is opaque to attackers who might want to compromise a specific user's path.
5. **Verifiable transparency.** Every binary running on PCC nodes is published in a transparency log. Independent researchers verify.

Reference: [Apple Security — Private Cloud Compute](https://security.apple.com/blog/private-cloud-compute/) and the [PCC verification guide](https://security.apple.com/documentation/private-cloud-compute/).

## Attestation flow — what happens per request

```
1. Device fetches the latest signed PCC node attestation bundle
   (CPU root cert, Secure Boot measurements, image hashes).
2. Device verifies the bundle against:
     - Apple's CA
     - the public Binary Transparency Log
     - a local list of known-good image hashes (updated by OS)
3. If verification passes, the device negotiates a session key with
   the specific node, encrypted to that node's hardware-attested
   public key.
4. Request payload is encrypted with the session key, shipped over TLS,
   and decrypted only inside the node's sealed memory.
5. Inference runs. Logits / response are encrypted back to the device's
   ephemeral key.
6. Session key is destroyed. Memory is zeroed. No log entry contains
   request content.
```

The cryptographic invariant: a node that is not running a published, transparency-logged image cannot complete the attestation handshake, so the device refuses to send the request.

## On-device vs PCC routing in Apple Intelligence

Foundation Models framework auto-routes between on-device (the ~3B model from Topic 07) and PCC (a larger Apple server model) using a **verifier model** approach:

```
       prompt
         |
         v
   [ on-device 3B ]  ----------+
         |                      |
         v                      v
   [ verifier ] - confidence high? -> ship the on-device answer
         |
         | confidence low / refusal / out-of-domain
         v
   [ PCC routing ] -> attestation -> server inference -> response
```

The verifier is an on-device classifier trained to predict whether the on-device model's answer is acceptable. Routing is a policy choice the developer can override (e.g., force on-device for sensitive prompts; force PCC for known-hard prompts). Apple's framework exposes this as `LanguageModelSession` configuration.

A practical implication for a local agent: even with Foundation Models in the loop, you should know which posture each request landed in and surface it to the user. Apple does this with the small "Apple Intelligence" indicator; your app should be at least as honest.

## Threat-model questions for **your** local stack

Use these to write the privacy section of `reports/local.md`:

1. **Where does each model run?** On-device (which framework: Foundation Models, MLX, llama.cpp), PCC, or cloud? List them.
2. **What tools does the agent call?** For each tool: does the call leave the device? If yes, what is the network endpoint, and what data goes to it?
3. **What does each layer log?** llama.cpp's server logs prompts by default. Ollama logs metadata. Your own agent loop probably logs more than you think. Audit.
4. **What persists on disk?** KV cache snapshots, session histories, dataset for fine-tuning. The threat model includes a stolen unlocked laptop.
5. **What permissions does the agent's process have?** Full Disk Access? Accessibility? Screen Recording? Each one widens the on-device threat surface.
6. **What does the user see?** A user who thinks "local" while a tool call ships text to a remote API has been misled, even if the model itself was local.

## A 2026 default for an honest privacy posture

```
Default routing for a "private" local agent:
  - Tier 1 (every prompt): on-device 3B via Foundation Models or local MLX 7B.
  - Tier 2 (verifier escalates):
      - if the user has opted in: PCC.
      - else: surface a "this needs cloud, continue?" prompt.
  - Tier 3 (user explicitly chose): cloud LLM, with a banner.
Logs stripped of prompt content by default; opt-in for debug mode.
Tools that leave the device: explicitly enumerated, surfaced in UI.
```

That posture is not maximalist privacy. Maximalist privacy is on-device-only with no network tools, and it is achievable. Most useful agents settle somewhere along the tier ladder, and the honest design is to make the tier visible.

## Common pitfalls

1. **Calling a stack "private" because the model is local while a web-search tool ships every query to Google.**
2. **Trusting "we don't train on your data" as equivalent to "your data never leaves our servers."** They are not the same.
3. **Storing fine-tuning datasets in plaintext on disk** when the whole point was to keep that data off cloud servers.
4. **Forgetting that the OS itself talks to the network.** Spotlight, iCloud, Mail, Siri — these are not your agent, but they share the device's threat surface.
5. **Treating PCC as identical to on-device.** It is much closer to on-device than cloud LLMs are, but it still requires trusting Apple's hardware design and transparency program. Document the difference.

## What to walk away with

- A clear three-posture model (on-device / PCC / cloud) and the precise guarantees of each.
- The PCC attestation flow at a level you can explain to a security reviewer.
- A written threat model for your `local-agent` that names every model, every tool, every log, and the posture each lands in.

## References

- Apple Security — Private Cloud Compute: https://security.apple.com/blog/private-cloud-compute/
- PCC verification documentation: https://security.apple.com/documentation/private-cloud-compute/
- PCC Security Guide (PDF, Apple): https://security.apple.com/documentation/private-cloud-compute/securityguide
- Apple Platform Security — Secure Enclave: https://support.apple.com/guide/security/secure-enclave-sec59b0b31ff/web
- WWDC25 — Bring on-device intelligence to your app with Foundation Models: https://developer.apple.com/videos/play/wwdc2025/286/
- Apple Intelligence privacy overview: https://www.apple.com/privacy/docs/Apple_Intelligence_and_Privacy.pdf
- Binary Transparency at Apple: https://security.apple.com/blog/private-cloud-compute/#transparency

# 16 — Privacy and Private Cloud Compute

## Files

- `CONCEPTS.md` — three privacy postures, what on-device does and doesn't buy, PCC architecture and attestation flow, on-device-vs-PCC routing in Apple Intelligence, threat-model questions for your stack.
- `pcc_attestation_walkthrough.py` — narrated stub of the PCC attestation handshake. Does **not** speak to real Apple servers; it walks the steps with mock keys so you can see the shape of the protocol and where each guarantee comes from.
- `threat_model_template.md` — fill-in-the-blanks template for the privacy section of `reports/local.md` (Project 4 deliverable).
- `threat_model.py` — interactive generator: walks the six threat-model questions and emits a Markdown section.
- `posture_audit.py` — probes running local model endpoints and configured tool URLs, labels each as on-device / Apple-hosted / cloud, and flags any data-leaving calls.

## Quickstart

Walk the attestation flow in the terminal:

```bash
python pcc_attestation_walkthrough.py
```

Open the threat-model template and fill it in for whatever your `local-agent` actually does:

```bash
$EDITOR threat_model_template.md
```

Or run the interactive generator and audit your live stack:

```bash
python threat_model.py --output threat_model.md
python posture_audit.py \
    --endpoints http://localhost:11434/v1 http://localhost:8000/v1 \
    --tool-urls https://api.duckduckgo.com https://api.openai.com
```

## Expected output

`pcc_attestation_walkthrough.py` prints each attestation step with the role each cryptographic primitive plays. Sample tail:

```
[5] node decrypts the session key inside its sealed memory only.
    -> Property: even Apple operations staff cannot read the session key.
[6] inference runs; response encrypted back to device's ephemeral key.
[7] session key destroyed; memory zeroed; no log carries the prompt.
    -> Property: stateless computation on personal data.

DONE. Five Apple PCC claims map to specific steps as follows:
  Stateless compute on personal data : steps 5, 7
  Enforceable guarantees             : steps 1, 2 (HW-rooted)
  No privileged runtime access       : step 5
  Non-targetability                  : step 1 (opaque node selection)
  Verifiable transparency            : step 2 (Binary Transparency Log)
```

## Try

- Read [Apple's actual PCC verification doc](https://security.apple.com/documentation/private-cloud-compute/) and compare the real attestation bundle structure to the simplified one in the walkthrough. Note the parts the simplified version skips (CPU boot measurement details, image manifest signing).
- Audit your own local stack with the threat-model template. For each tool the agent can call, write the network endpoint and the data shipped. The list usually has surprises.
- Pick three "AI privacy" marketing pages from any vendor and locate (or fail to locate) the equivalents of Apple's five PCC claims. The ones that don't have hardware-rooted stateless computation will be vague exactly where it matters.

## Where this goes

This is the last topic of Level 8. The output here is the privacy section of `reports/local.md` for Project 4. Level 9 (compiler tour) follows; the privacy posture you established here is what you defend when asked "ok, but is it really local?"

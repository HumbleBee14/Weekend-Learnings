"""Narrated walkthrough of the Apple Private Cloud Compute attestation flow.

This is a teaching stub. It does not contact any real PCC node. It uses mock
keys and a simplified message ordering to make the cryptographic structure
visible. Read alongside Apple's actual documentation:
  https://security.apple.com/blog/private-cloud-compute/
  https://security.apple.com/documentation/private-cloud-compute/

The point: see *which step* delivers each of Apple's five PCC properties.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass


def h(label: str, value: bytes) -> str:
    digest = hashlib.sha256(value).hexdigest()[:12]
    return f"{label}={digest}"


@dataclass
class AttestationBundle:
    """What a real PCC node ships during the handshake. Simplified."""
    node_pubkey: bytes              # hardware-attested per-node public key
    cpu_root_cert: bytes            # Apple CPU manufacturing cert
    secure_boot_measurement: bytes  # measurement of the boot chain
    image_hash: bytes               # hash of the running PCC software image
    transparency_proof: bytes       # inclusion proof in Binary Transparency Log
    signature: bytes                # signature over the above by node attestation key


def fake_attestation_bundle() -> AttestationBundle:
    return AttestationBundle(
        node_pubkey=os.urandom(32),
        cpu_root_cert=b"apple-cpu-root-cert-MOCK",
        secure_boot_measurement=hashlib.sha256(b"boot-chain-v2026.05").digest(),
        image_hash=hashlib.sha256(b"pcc-image-2026-05-01").digest(),
        transparency_proof=hashlib.sha256(b"binary-transparency-leaf-12345").digest(),
        signature=os.urandom(64),
    )


def device_known_good_hashes() -> set[bytes]:
    """The OS ships with (and updates) a list of known-good PCC image hashes
    that researchers have verified against the public transparency log."""
    return {hashlib.sha256(b"pcc-image-2026-05-01").digest()}


def step(n: int, msg: str, prop: str | None = None) -> None:
    print(f"[{n}] {msg}")
    if prop:
        print(f"    -> Property: {prop}")


def main() -> None:
    print("=" * 70)
    print("Private Cloud Compute attestation walkthrough (mocked)")
    print("=" * 70)
    print()

    # -----------------------------------------------------------------
    # Step 1: device fetches an attestation bundle from a candidate node.
    # The node is opaque to attackers; the device cannot pre-pick a node
    # and an attacker cannot route a target to a chosen node.
    # -----------------------------------------------------------------
    bundle = fake_attestation_bundle()
    step(
        1,
        f"device fetched an attestation bundle from a PCC node "
        f"({h('node_pubkey', bundle.node_pubkey)}, "
        f"{h('image', bundle.image_hash)}).",
        prop="Non-targetability — device did not choose the node.",
    )

    # -----------------------------------------------------------------
    # Step 2: device verifies the bundle. Three checks:
    #   (a) signature chain back to Apple CPU root.
    #   (b) image hash present in the public Binary Transparency Log.
    #   (c) image hash present in the device's locally-cached known-good set.
    # -----------------------------------------------------------------
    cpu_root_ok = bundle.cpu_root_cert == b"apple-cpu-root-cert-MOCK"
    transparency_ok = bundle.transparency_proof != b""   # mocked
    known_good_ok = bundle.image_hash in device_known_good_hashes()

    if not (cpu_root_ok and transparency_ok and known_good_ok):
        print("    ABORT: attestation failed; device refuses to send the request.")
        return

    step(
        2,
        f"device verified the bundle: CPU-root-chain ok, "
        f"transparency-log inclusion ok, image hash matches known-good set.",
        prop="Verifiable transparency — every PCC binary is publicly logged "
             "and independently audited.",
    )

    # -----------------------------------------------------------------
    # Step 3: device generates an ephemeral session key, encrypts it to
    # the node's hardware-attested public key. Only the sealed enclave
    # on that specific node can decrypt it.
    # -----------------------------------------------------------------
    session_key = secrets.token_bytes(32)
    encrypted_session_key = hashlib.sha256(
        bundle.node_pubkey + session_key
    ).digest()  # mock seal
    step(
        3,
        f"device sealed an ephemeral session key to the node's hardware key "
        f"({h('sealed', encrypted_session_key)}).",
        prop="Enforceable guarantees — only the node's secure hardware can "
             "unseal; not Apple operations, not the network.",
    )

    # -----------------------------------------------------------------
    # Step 4: device encrypts the prompt with the session key and ships it.
    # -----------------------------------------------------------------
    prompt = b"summarize this confidential meeting transcript: ..."
    encrypted_prompt = hashlib.sha256(session_key + prompt).digest()  # mock AEAD
    step(
        4,
        f"device encrypted the prompt under the session key and shipped it "
        f"({h('ciphertext', encrypted_prompt)}).",
    )

    # -----------------------------------------------------------------
    # Step 5: node unseals the session key inside sealed memory; decrypts
    # the prompt; runs inference. There is no SSH, no admin shell, no
    # mechanism for an Apple SRE to read this memory.
    # -----------------------------------------------------------------
    step(
        5,
        "node decrypts the session key inside its sealed memory only.",
        prop="No privileged runtime access — even Apple staff cannot read "
             "this memory.",
    )

    # -----------------------------------------------------------------
    # Step 6: response encrypted back to the device's ephemeral key.
    # -----------------------------------------------------------------
    response = b"meeting summary..."
    encrypted_response = hashlib.sha256(session_key + response).digest()
    step(
        6,
        f"inference runs; response encrypted back to the device's "
        f"ephemeral key ({h('response_ct', encrypted_response)}).",
    )

    # -----------------------------------------------------------------
    # Step 7: session key destroyed; memory zeroed; no log row contains
    # prompt content.
    # -----------------------------------------------------------------
    session_key = b"\x00" * 32
    step(
        7,
        "session key destroyed; memory zeroed; no log carries the prompt.",
        prop="Stateless computation on personal data.",
    )

    print()
    print("DONE. Five Apple PCC claims map to specific steps as follows:")
    print("  Stateless compute on personal data : steps 5, 7")
    print("  Enforceable guarantees             : steps 1, 2 (HW-rooted)")
    print("  No privileged runtime access       : step 5")
    print("  Non-targetability                  : step 1 (opaque node selection)")
    print("  Verifiable transparency            : step 2 (Binary Transparency Log)")
    print()
    print("Real protocol details (key derivation, AEAD choice, OHTTP relays")
    print("for IP unlinkability, image manifest signing) are in Apple's docs.")


if __name__ == "__main__":
    main()

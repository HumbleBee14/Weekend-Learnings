"""
Interactive threat-model writer for a local-agent stack.

Walks through the six questions from CONCEPTS.md and emits a Markdown
section you can drop into reports/local.md. Non-interactive default
answers are provided for a typical Mac local-agent setup so you can run
it once and then edit.
"""
from __future__ import annotations
import argparse
import textwrap
from datetime import date


SECTIONS = [
    ("models",
     "Where does each model run? (Foundation Models / MLX / llama.cpp / PCC / cloud)",
     "- chat: MLX (Qwen2.5-7B 4-bit, on-device)\n"
     "- autocomplete: MLX (Qwen2.5-Coder-1.5B, on-device)\n"
     "- embeddings: MLX (bge-m3, on-device)"),
    ("tools",
     "What tools does the agent call? Mark which leave the device.",
     "- read_file: on-device\n"
     "- edit_file: on-device\n"
     "- run_shell: on-device (sandboxed workdir)\n"
     "- web_fetch: LEAVES DEVICE (HTTPS to user-specified URL)"),
    ("logs",
     "What does each layer log, and where?",
     "- mlx_lm.server: stderr only, no prompt persistence\n"
     "- agent loop: ./logs/agent.jsonl (prompts + tool calls)\n"
     "- system: macOS unified log (no prompt content)"),
    ("persistence",
     "What persists on disk?",
     "- KV cache: in-memory only\n"
     "- session histories: ./sessions/*.jsonl (cleared on quit)\n"
     "- fine-tuning data: ./data/*.jsonl (encrypted disk; FileVault)"),
    ("permissions",
     "What permissions does the agent's process hold?",
     "- Full Disk Access: NO\n"
     "- Accessibility: NO\n"
     "- Screen Recording: NO\n"
     "- Network: YES (for web_fetch tool only)"),
    ("user_surface",
     "What does the user see about routing posture?",
     "- Banner shows 'on-device' for all model calls.\n"
     "- web_fetch surfaces a per-call confirmation with the URL.\n"
     "- No tier escalation to PCC or cloud in this build."),
]


def prompt(label: str, q: str, default: str) -> str:
    print(f"\n## {label}")
    print(q)
    print("Default:")
    print(textwrap.indent(default, "    "))
    print("Enter alternative text, or blank line to accept default. End with EOF (Ctrl-D).")
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    text = "\n".join(lines).strip()
    return text or default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="threat_model.md")
    ap.add_argument("--non-interactive", action="store_true",
                    help="emit defaults without prompting")
    args = ap.parse_args()

    answers = {}
    for label, q, default in SECTIONS:
        if args.non_interactive:
            answers[label] = default
            print(f"[non-interactive] {label}")
        else:
            answers[label] = prompt(label, q, default)

    out = [f"# Threat model — local-agent ({date.today().isoformat()})\n"]
    out.append("Three-posture framing: on-device / PCC / cloud. "
               "This document enumerates which posture every component lands in.\n")
    for label, _, _ in SECTIONS:
        out.append(f"## {label.replace('_', ' ').title()}\n")
        out.append(answers[label] + "\n")

    out.append("## Summary\n")
    out.append("This stack is on-device for all model inference. "
               "Network leaves the device only via explicitly enumerated tools, "
               "each gated by a per-call confirmation.\n")

    with open(args.output, "w") as f:
        f.write("\n".join(out))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

"""
Minimal local agentic loop.

Hits any OpenAI-compatible endpoint (Ollama, vLLM-MLX, mlx_lm.server,
LM Studio, llama-server). Three tools: read_file, edit_file (string
replace), run_shell (timeout-bounded). Streams output. Step-limited.

This is intentionally bare. It is the skeleton, not a Cursor clone.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import time
from pathlib import Path

from openai import OpenAI


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file (max 64 KB).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exactly-once `old` with `new` in a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command, 10 s timeout. Returns stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Call when the task is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def tool_read_file(path: str) -> str:
    p = Path(path)
    data = p.read_bytes()[:64 * 1024]
    return data.decode("utf-8", errors="replace")


def tool_edit_file(path: str, old: str, new: str) -> str:
    p = Path(path)
    text = p.read_text()
    if text.count(old) != 1:
        return f"error: `old` matched {text.count(old)} times (need exactly 1)"
    p.write_text(text.replace(old, new, 1))
    return "ok"


def tool_run_shell(cmd: str) -> str:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return (r.stdout + r.stderr)[:4096]
    except subprocess.TimeoutExpired:
        return "error: timeout"


DISPATCH = {
    "read_file": lambda a: tool_read_file(a["path"]),
    "edit_file": lambda a: tool_edit_file(a["path"], a["old"], a["new"]),
    "run_shell": lambda a: tool_run_shell(a["cmd"]),
}


def run(client: OpenAI, model: str, task: str, max_steps: int = 12) -> None:
    messages = [
        {"role": "system",
         "content": "You are a local coding agent. Use tools to complete the task. "
                    "Call `done` when finished."},
        {"role": "user", "content": task},
    ]
    t_start = time.perf_counter()
    for step in range(1, max_steps + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump())

        tcs = msg.tool_calls or []
        if not tcs:
            print(f"[step {step}] (no tool call) {msg.content[:200] if msg.content else ''}")
            break
        for tc in tcs:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            if name == "done":
                wall = time.perf_counter() - t_start
                print(f"[done] task complete in {step} steps, {wall:.1f}s wall")
                return
            print(f"[step {step}] tool={name} args={args}")
            out = DISPATCH.get(name, lambda a: "error: unknown tool")(args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": out,
            })
    print("[stop] step cap reached")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--max-steps", type=int, default=12)
    args = p.parse_args()
    client = OpenAI(base_url=args.base_url, api_key="-")
    run(client, args.model, args.task, args.max_steps)


if __name__ == "__main__":
    main()

"""
Posture auditor: probes local model endpoints and tool URLs, then labels
each by privacy posture (on-device / PCC / cloud) for the report.

The classifier is intentionally simple — local hosts get on-device,
*.apple.com paths that look like Foundation Models / PCC get PCC, the
rest get cloud. Edit CLOUD_KNOWN if you want richer detection.
"""
from __future__ import annotations
import argparse
import socket
from urllib.parse import urlparse

import httpx


LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}

CLOUD_KNOWN = {
    "api.openai.com": "cloud (OpenAI)",
    "api.anthropic.com": "cloud (Anthropic)",
    "generativelanguage.googleapis.com": "cloud (Google)",
    "api.groq.com": "cloud (Groq)",
    "api.deepseek.com": "cloud (DeepSeek)",
    "api.duckduckgo.com": "cloud (DuckDuckGo)",
}


def _is_local(host: str) -> bool:
    if host in LOCAL_HOSTS:
        return True
    try:
        ip = socket.gethostbyname(host)
    except OSError:
        return False
    return ip.startswith("127.") or ip.startswith("169.254.") or ip == "::1"


def classify(url: str) -> str:
    host = urlparse(url).hostname or ""
    if _is_local(host):
        return "on-device"
    if host.endswith("apple.com") or "icloud" in host:
        # Heuristic; real PCC traffic does not flow through user code directly,
        # but Apple-hosted endpoints worth flagging differently from third party.
        return "Apple-hosted (likely PCC if Foundation Models)"
    return CLOUD_KNOWN.get(host, "cloud (third party)")


def probe(url: str, timeout: float = 3.0) -> str:
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as c:
            r = c.get(url)
        return f"reach=ok ({r.status_code})"
    except Exception as e:
        return f"reach=fail ({type(e).__name__})"


def detect_engine(url: str) -> str:
    """Best-effort engine detection from the OpenAI-compatible /v1/models endpoint."""
    try:
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{url.rstrip('/')}/models")
        if r.status_code != 200:
            return "unknown"
        data = r.json()
        if "ollama" in r.text.lower():
            return "Ollama"
        if any("mlx" in m.get("id", "").lower() for m in data.get("data", [])):
            return "mlx_lm.server"
        return "OpenAI-compatible"
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", nargs="*", default=[],
                    help="Local model HTTP endpoints (e.g. http://localhost:11434/v1)")
    ap.add_argument("--tool-urls", nargs="*", default=[],
                    help="URLs the agent's tools call")
    args = ap.parse_args()

    print("=== Local model endpoints ===")
    on_dev_models = 0
    for url in args.endpoints:
        posture = classify(url)
        engine = detect_engine(url)
        reach = probe(f"{url.rstrip('/')}/models")
        print(f"  {url:40s}  posture={posture}  engine={engine}  {reach}")
        if posture == "on-device":
            on_dev_models += 1

    print("\n=== Tool URLs ===")
    leaving = 0
    for url in args.tool_urls:
        posture = classify(url)
        reach = probe(url)
        flag = "[data leaves device]" if posture != "on-device" else ""
        print(f"  {url:40s}  posture={posture:30s}  {reach}  {flag}")
        if posture != "on-device":
            leaving += 1

    print(f"\nVerdict: {on_dev_models} model(s) on-device, "
          f"{leaving} tool(s) leave the device.")
    if leaving:
        print("Surface this in your UI; it's the difference between "
              "'private agent' and 'agent with cloud tools.'")


if __name__ == "__main__":
    main()

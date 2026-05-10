"""
Test structured-output grammar masking via vLLM's OpenAI-compatible API.

Setup:
    pip install vllm openai
    vllm serve Qwen/Qwen2.5-1.5B-Instruct  &
    python test_grammar.py
"""

import json
import time

import openai


client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT = "Recommend a restaurant in Berlin with rating, cuisine, and price range."

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "city": {"type": "string"},
        "rating": {"type": "number", "minimum": 0, "maximum": 5},
        "cuisine": {"type": "string"},
        "price_range": {"type": "string", "enum": ["$", "$$", "$$$", "$$$$"]},
    },
    "required": ["name", "city", "rating", "cuisine", "price_range"],
}


def call(use_schema: bool):
    extra = {}
    if use_schema:
        extra["response_format"] = {
            "type": "json_schema",
            "json_schema": {"schema": SCHEMA},
        }

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        max_tokens=200,
        temperature=0.7,
        **extra,
    )
    elapsed = time.perf_counter() - t0
    content = response.choices[0].message.content

    # Try to parse as JSON
    try:
        parsed = json.loads(content)
        valid = True
    except json.JSONDecodeError:
        parsed = None
        valid = False

    return elapsed, content, valid, parsed


def main():
    print("=== Without grammar masking ===")
    n_unconstrained = 5
    valid_count = 0
    times = []
    for i in range(n_unconstrained):
        t, content, valid, _ = call(use_schema=False)
        times.append(t)
        if valid:
            valid_count += 1
        print(f"  [{i + 1}/{n_unconstrained}] valid_json={valid}  time={t:.2f}s")
        print(f"      {content[:100]}...")

    print(f"\n  → {valid_count}/{n_unconstrained} produced valid JSON")
    print(f"  → median time: {sorted(times)[len(times) // 2]:.2f}s")

    print("\n=== With JSON schema grammar masking ===")
    n_constrained = 5
    valid_count = 0
    schema_compliant = 0
    times = []
    for i in range(n_constrained):
        t, content, valid, parsed = call(use_schema=True)
        times.append(t)
        if valid:
            valid_count += 1
            # Check schema compliance
            if all(k in parsed for k in SCHEMA["required"]):
                schema_compliant += 1
        print(f"  [{i + 1}/{n_constrained}] valid_json={valid}  time={t:.2f}s")
        if parsed:
            print(f"      {json.dumps(parsed, indent=None)[:120]}")

    print(f"\n  → {valid_count}/{n_constrained} produced valid JSON (should be 100%)")
    print(f"  → {schema_compliant}/{n_constrained} schema-compliant")
    print(f"  → median time: {sorted(times)[len(times) // 2]:.2f}s")

    print("\nNotes:")
    print("- Without grammar: model often produces 'natural language with embedded JSON'.")
    print("  Parse failure rate is the headline gotcha for production.")
    print("- With grammar: 100% parseable, schema-compliant by construction.")
    print("- Time delta: small, often within noise. xgrammar's overhead is near-zero.")


if __name__ == "__main__":
    main()

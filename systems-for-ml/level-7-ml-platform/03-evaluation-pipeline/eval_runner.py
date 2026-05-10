"""
Eval runner + regression gate for mini-platform.

Reads candidate scores from a JSON file, fetches previous `serving` scores
from the registry SQLite (Topic 04), evaluates the gate, writes the verdict.

This script does NOT run lm-eval-harness for you. In real use:

    lm_eval --model vllm --model_args pretrained=$CKPT \\
            --tasks mmlu,gsm8k,humaneval \\
            --output_path results/$JOB.json

Then:

    python eval_runner.py gate \\
        --candidate-id $CKPT_ID \\
        --scores results/$JOB.json \\
        --gate gate.yaml
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None

REGISTRY_DB = Path(__file__).parent.parent / "04-model-registry" / "registry.sqlite"


def load_gate(path: str) -> dict:
    text = Path(path).read_text()
    if yaml is not None:
        return yaml.safe_load(text)
    # Fallback: JSON also accepted.
    return json.loads(text)


def fetch_prev_serving_scores(model_name: str) -> dict:
    if not REGISTRY_DB.exists():
        return {}
    con = sqlite3.connect(REGISTRY_DB)
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT eval_scores_json FROM models "
        "WHERE name=? AND status='serving' "
        "ORDER BY version DESC LIMIT 1",
        (model_name,),
    ).fetchone()
    con.close()
    if not row or not row["eval_scores_json"]:
        return {}
    return json.loads(row["eval_scores_json"])


def evaluate_rule(rule: str, candidate: float, prev: float | None) -> bool:
    """
    Supported rules:
        ge_rel(prev, X)   candidate >= prev * X     (regression floor)
        le_rel(prev, X)   candidate <= prev * X     (suspicious-gain ceiling)
        ge_abs(X)         candidate >= X            (absolute floor)
    """
    rule = rule.strip()
    if rule.startswith("ge_rel(prev,"):
        x = float(rule[rule.index(",") + 1 : rule.rindex(")")])
        if prev is None:
            return True  # first run; allow
        return candidate >= prev * x
    if rule.startswith("le_rel(prev,"):
        x = float(rule[rule.index(",") + 1 : rule.rindex(")")])
        if prev is None:
            return True
        return candidate <= prev * x
    if rule.startswith("ge_abs("):
        x = float(rule[rule.index("(") + 1 : rule.rindex(")")])
        return candidate >= x
    raise ValueError(f"unknown rule: {rule}")


def run_gate(candidate_scores: dict, prev_scores: dict, gate: dict) -> dict:
    results = []
    passes = 0
    for rule_def in gate["rules"]:
        bench = rule_def["benchmark"]
        metric = rule_def["metric"]
        rule = rule_def["rule"]

        cand = candidate_scores.get(bench, {}).get(metric)
        prev = prev_scores.get(bench, {}).get(metric)

        if cand is None:
            ok = False
            note = f"missing candidate score for {bench}/{metric}"
        else:
            ok = evaluate_rule(rule, cand, prev)
            note = f"cand={cand:.4f} prev={prev if prev is None else f'{prev:.4f}'} rule={rule}"

        results.append({"benchmark": bench, "metric": metric, "pass": ok, "note": note})
        if ok:
            passes += 1

    required = gate.get("required_pass", len(gate["rules"]))
    verdict = "approved" if passes >= required else "rejected"
    return {"verdict": verdict, "passed": passes, "required": required, "rules": results}


def cmd_gate(args):
    candidate = json.loads(Path(args.scores).read_text())
    gate = load_gate(args.gate)
    prev = fetch_prev_serving_scores(args.model_name)
    result = run_gate(candidate, prev, gate)
    print(json.dumps(result, indent=2))
    if args.write_status and REGISTRY_DB.exists():
        con = sqlite3.connect(REGISTRY_DB)
        con.execute(
            "UPDATE models SET status=?, eval_scores_json=? WHERE model_id=?",
            (result["verdict"], json.dumps(candidate), args.candidate_id),
        )
        con.commit()
        con.close()
    sys.exit(0 if result["verdict"] == "approved" else 2)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("--candidate-id", required=True)
    g.add_argument("--model-name", required=True)
    g.add_argument("--scores", required=True, help="path to candidate scores JSON")
    g.add_argument("--gate", required=True, help="path to gate.yaml")
    g.add_argument("--write-status", action="store_true",
                   help="write verdict back to registry")
    g.set_defaults(func=cmd_gate)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

"""
mini-platform model registry.

Five-state status machine. SQLite-backed. Atomic promote/rollback.
At most one `serving` row per model name (DB-enforced via partial unique index).

Usage:
    python registry.py register --name minigpt --version v0.1 --path ./models/v0.1
    python registry.py set-eval <model_id> '{"mmlu": {"acc": 0.65}}'
    python registry.py approve <model_id>
    python registry.py promote <model_id>
    python registry.py rollback --name minigpt
    python registry.py list --name minigpt
"""

import argparse
import json
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "registry.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id          TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    version           TEXT NOT NULL,
    path              TEXT NOT NULL,
    format            TEXT NOT NULL DEFAULT 'safetensors',
    quantization      TEXT,
    base_model_id     TEXT,
    adapter_type      TEXT,
    eval_scores_json  TEXT,
    status            TEXT NOT NULL DEFAULT 'staged',
    parent_model_id   TEXT,
    created_at        REAL NOT NULL,
    promoted_at       REAL,
    retired_at        REAL,
    metadata_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_models_name_status ON models(name, status);
CREATE UNIQUE INDEX IF NOT EXISTS idx_serving_per_name
    ON models(name) WHERE status='serving';
"""

VALID = {"staged", "eval", "approved", "rejected", "serving", "retired", "canary"}


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript(SCHEMA)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def register(name, version, path, **kw) -> str:
    model_id = uuid.uuid4().hex[:12]
    with db() as con:
        # parent = current serving for rollback chain.
        prev = con.execute(
            "SELECT model_id FROM models WHERE name=? AND status='serving'", (name,)
        ).fetchone()
        con.execute(
            "INSERT INTO models(model_id,name,version,path,format,quantization,"
            "base_model_id,adapter_type,parent_model_id,created_at,metadata_json,status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                model_id, name, version, path,
                kw.get("format", "safetensors"),
                kw.get("quantization"),
                kw.get("base_model_id"),
                kw.get("adapter_type"),
                prev["model_id"] if prev else None,
                time.time(),
                json.dumps(kw.get("metadata", {})),
                "staged",
            ),
        )
    return model_id


def set_status(model_id: str, status: str):
    if status not in VALID:
        raise ValueError(f"bad status: {status}")
    with db() as con:
        con.execute("UPDATE models SET status=? WHERE model_id=?", (status, model_id))


def set_eval_scores(model_id: str, scores: dict):
    with db() as con:
        con.execute(
            "UPDATE models SET eval_scores_json=? WHERE model_id=?",
            (json.dumps(scores), model_id),
        )


def promote(model_id: str):
    """approved -> serving, atomic flip with previous serving -> retired."""
    now = time.time()
    with db() as con:
        row = con.execute("SELECT name, status FROM models WHERE model_id=?",
                          (model_id,)).fetchone()
        if not row:
            raise ValueError("not found")
        if row["status"] not in {"approved", "canary"}:
            raise ValueError(f"can only promote approved|canary, not {row['status']}")
        # Single transaction.
        con.execute(
            "UPDATE models SET status='retired', retired_at=? "
            "WHERE name=? AND status='serving'",
            (now, row["name"]),
        )
        con.execute(
            "UPDATE models SET status='serving', promoted_at=? WHERE model_id=?",
            (now, model_id),
        )


def rollback(name: str):
    """Most recent retired -> serving, current serving -> retired."""
    now = time.time()
    with db() as con:
        cur = con.execute(
            "SELECT model_id FROM models WHERE name=? AND status='serving'", (name,)
        ).fetchone()
        prev = con.execute(
            "SELECT model_id FROM models WHERE name=? AND status='retired' "
            "ORDER BY retired_at DESC LIMIT 1",
            (name,),
        ).fetchone()
        if not prev:
            raise ValueError("nothing to roll back to")
        if cur:
            con.execute(
                "UPDATE models SET status='retired', retired_at=? WHERE model_id=?",
                (now, cur["model_id"]),
            )
        con.execute(
            "UPDATE models SET status='serving', promoted_at=? WHERE model_id=?",
            (now, prev["model_id"]),
        )


def get_serving(name: str) -> dict | None:
    with db() as con:
        row = con.execute(
            "SELECT * FROM models WHERE name=? AND status='serving'", (name,)
        ).fetchone()
    return dict(row) if row else None


def list_models(name: str | None = None) -> list[dict]:
    with db() as con:
        if name:
            rows = con.execute(
                "SELECT * FROM models WHERE name=? ORDER BY created_at DESC", (name,)
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM models ORDER BY created_at DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("register")
    r.add_argument("--name", required=True)
    r.add_argument("--version", required=True)
    r.add_argument("--path", required=True)
    r.add_argument("--format", default="safetensors")
    r.add_argument("--quantization")
    r.add_argument("--base-model-id")
    r.add_argument("--adapter-type")

    s = sub.add_parser("set-eval"); s.add_argument("model_id"); s.add_argument("scores_json")
    s = sub.add_parser("set-status"); s.add_argument("model_id"); s.add_argument("status")
    s = sub.add_parser("approve"); s.add_argument("model_id")
    s = sub.add_parser("reject"); s.add_argument("model_id")
    s = sub.add_parser("promote"); s.add_argument("model_id")
    s = sub.add_parser("rollback"); s.add_argument("--name", required=True)
    s = sub.add_parser("serving"); s.add_argument("--name", required=True)
    s = sub.add_parser("list"); s.add_argument("--name")

    args = p.parse_args()

    if args.cmd == "register":
        print(register(
            args.name, args.version, args.path,
            format=args.format, quantization=args.quantization,
            base_model_id=args.base_model_id, adapter_type=args.adapter_type,
        ))
    elif args.cmd == "set-eval":
        set_eval_scores(args.model_id, json.loads(args.scores_json))
    elif args.cmd == "set-status":
        set_status(args.model_id, args.status)
    elif args.cmd == "approve":
        set_status(args.model_id, "approved")
    elif args.cmd == "reject":
        set_status(args.model_id, "rejected")
    elif args.cmd == "promote":
        promote(args.model_id)
    elif args.cmd == "rollback":
        rollback(args.name)
    elif args.cmd == "serving":
        print(json.dumps(get_serving(args.name), indent=2, default=str))
    elif args.cmd == "list":
        for m in list_models(args.name):
            print(f"{m['model_id']}  {m['name']:<12}  {m['version']:<12}  "
                  f"{m['status']:<10}  {m.get('quantization') or '':<10}  {m['path']}")


if __name__ == "__main__":
    main()

"""
Minimum viable training job scheduler for mini-platform.

Five ops: submit, status, list, cancel, retry.
SQLite-backed. Subprocess-based. PID re-attaches on restart.

Run:
    python scheduler.py submit "sleep 5 && echo ok"
    python scheduler.py list
    python scheduler.py status <job_id>

Then induce a failure:
    python scheduler.py submit "sleep 1 && exit 1"
    python scheduler.py list   # observe FAILED
"""

import argparse
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent / "jobs.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT PRIMARY KEY,
    cmd             TEXT NOT NULL,
    status          TEXT NOT NULL,
    pid             INTEGER,
    started_at      REAL,
    finished_at     REAL,
    exit_code       INTEGER,
    retry_count     INTEGER NOT NULL DEFAULT 0,
    max_retries     INTEGER NOT NULL DEFAULT 3,
    last_step       INTEGER,
    last_loss       REAL,
    checkpoint_path TEXT,
    metadata_json   TEXT
);
"""


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def pid_alive(pid: int) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def submit(cmd: str, max_retries: int = 3, metadata: dict | None = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    with db() as con:
        con.execute(
            "INSERT INTO jobs(job_id, cmd, status, max_retries, metadata_json) "
            "VALUES (?,?,?,?,?)",
            (job_id, cmd, "PENDING", max_retries, json.dumps(metadata or {})),
        )
    return job_id


def _spawn(con, row) -> None:
    """Start a PENDING job. Caller holds the db connection."""
    cmd = row["cmd"]
    proc = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    con.execute(
        "UPDATE jobs SET status=?, pid=?, started_at=? WHERE job_id=?",
        ("RUNNING", proc.pid, time.time(), row["job_id"]),
    )


def reconcile() -> dict:
    """
    Walk the table once. Promote PENDING -> RUNNING. Reap dead RUNNING.
    Idempotent. Safe to call from a cron / loop / restart.
    """
    summary = {"started": 0, "finished": 0, "failed": 0, "retried": 0}
    with db() as con:
        # 1. Reap RUNNING whose PIDs are gone.
        for row in con.execute("SELECT * FROM jobs WHERE status='RUNNING'").fetchall():
            if not pid_alive(row["pid"]):
                # We don't have exit code post-hoc on a detached process; mark FAILED
                # if no checkpoint advanced; DONE if a finish-marker file exists.
                done_marker = Path(f"/tmp/mini_platform_done_{row['job_id']}")
                if done_marker.exists():
                    con.execute(
                        "UPDATE jobs SET status=?, finished_at=?, exit_code=0 "
                        "WHERE job_id=?",
                        ("DONE", time.time(), row["job_id"]),
                    )
                    done_marker.unlink(missing_ok=True)
                    summary["finished"] += 1
                else:
                    if row["retry_count"] < row["max_retries"]:
                        con.execute(
                            "UPDATE jobs SET status=?, retry_count=?, pid=NULL "
                            "WHERE job_id=?",
                            ("PENDING", row["retry_count"] + 1, row["job_id"]),
                        )
                        summary["retried"] += 1
                    else:
                        con.execute(
                            "UPDATE jobs SET status=?, finished_at=? WHERE job_id=?",
                            ("FAILED", time.time(), row["job_id"]),
                        )
                        summary["failed"] += 1

        # 2. Start PENDING.
        for row in con.execute(
            "SELECT * FROM jobs WHERE status='PENDING' ORDER BY rowid"
        ).fetchall():
            _spawn(con, row)
            summary["started"] += 1
    return summary


def status(job_id: str) -> dict | None:
    with db() as con:
        row = con.execute(
            "SELECT * FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def list_jobs() -> list[dict]:
    with db() as con:
        rows = con.execute(
            "SELECT job_id, status, pid, retry_count, cmd FROM jobs "
            "ORDER BY started_at DESC NULLS LAST"
        ).fetchall()
    return [dict(r) for r in rows]


def cancel(job_id: str) -> bool:
    with db() as con:
        row = con.execute(
            "SELECT pid, status FROM jobs WHERE job_id=?", (job_id,)
        ).fetchone()
        if not row:
            return False
        if row["status"] == "RUNNING" and pid_alive(row["pid"]):
            try:
                os.killpg(os.getpgid(row["pid"]), signal.SIGTERM)
            except ProcessLookupError:
                pass
        con.execute(
            "UPDATE jobs SET status='CANCELLED', finished_at=? WHERE job_id=?",
            (time.time(), job_id),
        )
    return True


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit"); s.add_argument("command"); s.add_argument("--max-retries", type=int, default=3)
    sub.add_parser("reconcile")
    s = sub.add_parser("status"); s.add_argument("job_id")
    sub.add_parser("list")
    s = sub.add_parser("cancel"); s.add_argument("job_id")
    sub.add_parser("loop")  # poll every 2s

    args = p.parse_args()

    if args.cmd == "submit":
        jid = submit(args.command, max_retries=args.max_retries)
        reconcile()
        print(jid)
    elif args.cmd == "reconcile":
        print(json.dumps(reconcile(), indent=2))
    elif args.cmd == "status":
        st = status(args.job_id)
        print(json.dumps(st, indent=2, default=str) if st else "not found")
    elif args.cmd == "list":
        for j in list_jobs():
            print(f"{j['job_id']}  {j['status']:<10}  pid={j['pid']}  retries={j['retry_count']}  cmd={j['cmd'][:60]}")
    elif args.cmd == "cancel":
        print("cancelled" if cancel(args.job_id) else "not found")
    elif args.cmd == "loop":
        print("Reconciling every 2s. Ctrl-C to stop.")
        while True:
            r = reconcile()
            if any(r.values()):
                print(time.strftime("%H:%M:%S"), r)
            time.sleep(2)


if __name__ == "__main__":
    main()

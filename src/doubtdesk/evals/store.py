from doubtdesk.config import settings
import json
import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = settings.evals_db


def initialize_evals_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS eval_runs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                mode TEXT NOT NULL,
                progress INTEGER NOT NULL DEFAULT 0,
                total INTEGER NOT NULL,
                result_json TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)


def mark_interrupted(db_path: Path = DB_PATH):
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("""
            UPDATE eval_runs SET status='interrupted', error='Server restarted during evaluation',
            updated_at=CURRENT_TIMESTAMP WHERE status IN ('queued', 'running')
        """)


def create_run(run_id: str, mode: str, total: int, db_path: Path = DB_PATH):
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "INSERT INTO eval_runs(id,status,mode,total) VALUES (?, 'queued', ?, ?)",
            (run_id, mode, total),
        )


def update_run(run_id: str, *, status=None, progress=None, result=None, error=None, db_path: Path = DB_PATH):
    changes = []
    params = []
    for key, value in (("status", status), ("progress", progress), ("result_json", json.dumps(result) if result is not None else None), ("error", error)):
        if value is not None:
            changes.append(f"{key} = ?")
            params.append(value)
    changes.append("updated_at = CURRENT_TIMESTAMP")
    params.append(run_id)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(f"UPDATE eval_runs SET {', '.join(changes)} WHERE id = ?", params)


def get_run(run_id: str, db_path: Path = DB_PATH):
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM eval_runs WHERE id = ?", (run_id,)).fetchone()
    if not row:
        return None
    result = dict(row)
    result["results"] = json.loads(result.pop("result_json"))
    return result

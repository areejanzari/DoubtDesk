from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from doubtdesk.db import TIMESTAMP_FORMAT, connection


def mark_interrupted():
    with connection() as conn:
        conn.execute("""
            UPDATE eval_runs SET status='interrupted', error='Server restarted during evaluation',
            updated_at=now() WHERE status IN ('queued', 'running')
        """)


def create_run(run_id: str, mode: str, total: int):
    with connection() as conn:
        conn.execute(
            "INSERT INTO eval_runs (id, status, mode, total) VALUES (%s, 'queued', %s, %s)",
            (run_id, mode, total),
        )


def update_run(run_id: str, *, status=None, progress=None, result=None, error=None):
    changes = []
    params = []
    for key, value in (("status", status), ("progress", progress), ("results", Jsonb(result) if result is not None else None), ("error", error)):
        if value is not None:
            changes.append(f"{key} = %s")
            params.append(value)
    changes.append("updated_at = now()")
    params.append(run_id)
    with connection() as conn:
        conn.execute(f"UPDATE eval_runs SET {', '.join(changes)} WHERE id = %s", params)


def get_run(run_id: str):
    with connection() as conn:
        return conn.cursor(row_factory=dict_row).execute(
            f"""
            SELECT id, status, mode, progress, total, error, results,
                   to_char(created_at AT TIME ZONE 'UTC', '{TIMESTAMP_FORMAT}') AS created_at,
                   to_char(updated_at AT TIME ZONE 'UTC', '{TIMESTAMP_FORMAT}') AS updated_at
            FROM eval_runs WHERE id = %s
            """,
            (run_id,),
        ).fetchone()

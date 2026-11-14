from doubtdesk.config import settings
import sqlite3
from pathlib import Path

DB_PATH = settings.analytics_db


def initialize_analytics_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                conversation_id TEXT,
                operation TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_tokens INTEGER,
                output_tokens INTEGER,
                thinking_tokens INTEGER,
                cached_tokens INTEGER,
                total_tokens INTEGER,
                latency_ms INTEGER,
                status TEXT NOT NULL,
                error_type TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        columns = {row[1] for row in conn.execute("PRAGMA table_info(ai_usage)")}
        if "provider" not in columns:
            conn.execute("ALTER TABLE ai_usage ADD COLUMN provider TEXT NOT NULL DEFAULT 'unknown'")
            conn.execute("UPDATE ai_usage SET provider = 'gemini' WHERE lower(model) LIKE 'gemini%'")
            conn.execute("UPDATE ai_usage SET provider = 'openai' WHERE lower(model) LIKE 'gpt-%'")
        if "pacer_wait_ms" not in columns:
            conn.execute("ALTER TABLE ai_usage ADD COLUMN pacer_wait_ms INTEGER")

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_usage_request
            ON ai_usage(request_id)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_ai_usage_created
            ON ai_usage(created_at)
            """
        )
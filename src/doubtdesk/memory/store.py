import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/conversations.db")


def initialize_memory_db(db_path: Path = DB_PATH):
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                doc_ids TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_exchanges_conversation
            ON exchanges (conversation_id, id)
        """)


def save_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    doc_ids: list[str],
    status: str,
    db_path: Path = DB_PATH,
):
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO exchanges
                (conversation_id, question, answer, doc_ids, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                question,
                answer,
                json.dumps(doc_ids),
                status,
            ),
        )


def get_recent_exchanges(
    conversation_id: str,
    limit: int = 3,
    db_path: Path = DB_PATH,
) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT question, answer, doc_ids, status
            FROM exchanges
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (conversation_id, limit),
        ).fetchall()

    return [
        {
            "question": row["question"],
            "answer": row["answer"],
            "doc_ids": json.loads(row["doc_ids"]),
            "status": row["status"],
        }
        for row in reversed(rows)
    ]

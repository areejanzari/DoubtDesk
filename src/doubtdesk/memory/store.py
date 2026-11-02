import json
import sqlite3
from pathlib import Path
from contextlib import closing

DB_PATH = Path("data/conversations.db")


def initialize_memory_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversations_user
            ON conversations(user_id)
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                doc_ids TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exchanges_conversation
            ON exchanges(conversation_id, id)
            """
        )



def save_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    doc_ids: list[str],
    status: str,
    db_path: Path = DB_PATH,
):
    with closing(sqlite3.connect(db_path)) as conn, conn:
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
    with closing(sqlite3.connect(db_path)) as conn, conn:
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


def claim_or_check_conversation(
    conversation_id: str,
    user_id: str,
    db_path: Path = DB_PATH,
) -> bool:
    with closing(sqlite3.connect(db_path, timeout=15)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            """
            SELECT user_id
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()

        if row is not None:
            return row[0] == user_id

        legacy_exchange = conn.execute(
            """
            SELECT 1
            FROM exchanges
            WHERE conversation_id = ?
            LIMIT 1
            """,
            (conversation_id,),
        ).fetchone()

        if legacy_exchange is not None:
            return False

        conn.execute(
            """
            INSERT INTO conversations (
                conversation_id,
                user_id
            )
            VALUES (?, ?)
            """,
            (conversation_id, user_id),
        )

    return True


def get_user_recent_exchanges(
    conversation_id: str,
    user_id: str,
    limit: int = 3,
    db_path: Path = DB_PATH,
) -> list[dict]:
    if limit < 1:
        raise ValueError("Limit must be positive.")

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                e.question,
                e.answer,
                e.doc_ids,
                e.status
            FROM exchanges AS e
            JOIN conversations AS c
                ON e.conversation_id = c.conversation_id
            WHERE e.conversation_id = ?
              AND c.user_id = ?
            ORDER BY e.id DESC
            LIMIT ?
            """,
            (conversation_id, user_id, limit),
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

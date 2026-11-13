import json
import sqlite3
from contextlib import closing
from pathlib import Path
from time import time_ns

DB_PATH = Path("data/conversations.db")
MAX_CONVERSATIONS_PER_USER = 30
MAX_CONVERSATION_TITLE_LENGTH = 80


class ConversationLimitReached(ValueError):
    pass


def _title_from_question(question: str) -> str:
    title = " ".join(question.split())
    return title[:MAX_CONVERSATION_TITLE_LENGTH].rstrip() or "New chat"


def initialize_memory_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed_ns INTEGER NOT NULL DEFAULT 0,
                title TEXT NOT NULL DEFAULT 'New chat',
                custom_title INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cols = {row[1] for row in conn.execute("PRAGMA table_info(conversations)")}
        if "last_accessed_ns" not in cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN last_accessed_ns INTEGER NOT NULL DEFAULT 0"
            )
        if "title" not in cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN title TEXT NOT NULL DEFAULT 'New chat'"
            )
        if "custom_title" not in cols:
            conn.execute(
                "ALTER TABLE conversations ADD COLUMN custom_title INTEGER NOT NULL DEFAULT 0"
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evicted_conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                evicted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
            CREATE INDEX IF NOT EXISTS idx_conversations_activity
            ON conversations(user_id, last_accessed_ns DESC)
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
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                formatted_answer TEXT,
                sources_json TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        exchange_cols = {row[1] for row in conn.execute("PRAGMA table_info(exchanges)")}
        if "formatted_answer" not in exchange_cols:
            conn.execute("ALTER TABLE exchanges ADD COLUMN formatted_answer TEXT")
        if "sources_json" not in exchange_cols:
            conn.execute(
                "ALTER TABLE exchanges ADD COLUMN sources_json TEXT NOT NULL DEFAULT '[]'"
            )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_exchanges_conversation
            ON exchanges(conversation_id, id)
            """
        )
        conn.execute(
            """
            UPDATE conversations
            SET title = SUBSTR(
                (SELECT e.question FROM exchanges e
                 WHERE e.conversation_id = conversations.conversation_id
                 ORDER BY e.id ASC LIMIT 1), 1, ?
            )
            WHERE custom_title = 0
              AND title = 'New chat'
              AND EXISTS (
                SELECT 1 FROM exchanges e
                WHERE e.conversation_id = conversations.conversation_id
              )
            """,
            (MAX_CONVERSATION_TITLE_LENGTH,),
        )


def save_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    doc_ids: list[str],
    status: str,
    db_path: Path = DB_PATH,
    *,
    user_id: str | None = None,
    formatted_answer: str | None = None,
    sources: list[dict] | None = None,
):
    with closing(sqlite3.connect(db_path, timeout=15)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        if user_id is not None:
            owner = conn.execute(
                "SELECT user_id FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            if owner is None or owner[0] != user_id:
                raise ValueError("Conversation not found.")
        cursor = conn.execute(
            """
            INSERT INTO exchanges
                (conversation_id, question, answer, doc_ids, status, formatted_answer, sources_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                conversation_id,
                question,
                answer,
                json.dumps(doc_ids),
                status,
                formatted_answer,
                json.dumps(sources or []),
            ),
        )
        conn.execute(
            """
            UPDATE conversations
            SET last_accessed_ns = ?,
                title = CASE WHEN custom_title = 0 AND NOT EXISTS (
                    SELECT 1 FROM exchanges
                    WHERE conversation_id = ? AND id < ?
                ) THEN ? ELSE title END
            WHERE conversation_id = ?
            """,
            (time_ns(), conversation_id, cursor.lastrowid, _title_from_question(question), conversation_id),
        )
        return cursor.lastrowid


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
    max_conversations: int = MAX_CONVERSATIONS_PER_USER,
) -> bool:
    if max_conversations < 1:
        raise ValueError("Maximum conversation count must be positive.")
    with closing(sqlite3.connect(db_path, timeout=15)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        if row is not None:
            if row[0] != user_id:
                return False
            return True

        if conn.execute(
            "SELECT 1 FROM evicted_conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone():
            return False

        if conn.execute(
            "SELECT 1 FROM exchanges WHERE conversation_id = ? LIMIT 1",
            (conversation_id,),
        ).fetchone():
            return False

        count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        if count >= max_conversations:
            raise ConversationLimitReached(
                f"You have reached the {max_conversations}-conversation limit. Delete a conversation before starting a new one."
            )
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, user_id, last_accessed_ns)
            VALUES (?, ?, ?)
            """,
            (conversation_id, user_id, time_ns()),
        )
    return True


def list_user_conversations(user_id: str, db_path: Path = DB_PATH) -> list[dict]:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT c.conversation_id, c.title, c.created_at, c.last_accessed_ns,
                   COUNT(e.id) AS message_count
            FROM conversations c
            LEFT JOIN exchanges e ON e.conversation_id = c.conversation_id
            WHERE c.user_id = ?
            GROUP BY c.conversation_id
            ORDER BY c.last_accessed_ns DESC, c.created_at DESC, c.conversation_id DESC
            """,
            (user_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def get_user_conversation(conversation_id: str, user_id: str, db_path: Path = DB_PATH) -> dict | None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT conversation_id, title, created_at, last_accessed_ns
            FROM conversations WHERE conversation_id = ? AND user_id = ?
            """,
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return None
        exchanges = conn.execute(
            """
            SELECT id, question, answer, COALESCE(formatted_answer, answer) AS formatted_answer,
                   sources_json, doc_ids, status, created_at
            FROM exchanges WHERE conversation_id = ? ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
        return {
            **dict(row),
            "messages": [
                {
                    "id": e["id"],
                    "question": e["question"],
                    "answer": e["answer"],
                    "formatted_answer": e["formatted_answer"],
                    "sources": json.loads(e["sources_json"] or "[]"),
                    "doc_ids": json.loads(e["doc_ids"]),
                    "status": e["status"],
                    "created_at": e["created_at"],
                }
                for e in exchanges
            ],
        }


def rename_user_conversation(conversation_id: str, user_id: str, title: str, db_path: Path = DB_PATH) -> bool:
    title = " ".join(title.split())
    if not title or len(title) > MAX_CONVERSATION_TITLE_LENGTH:
        raise ValueError(f"Title must be between 1 and {MAX_CONVERSATION_TITLE_LENGTH} characters.")
    with closing(sqlite3.connect(db_path)) as conn, conn:
        cursor = conn.execute(
            """
            UPDATE conversations SET title = ?, custom_title = 1
            WHERE conversation_id = ? AND user_id = ?
            """,
            (title, conversation_id, user_id),
        )
        return cursor.rowcount > 0


def delete_user_conversation(conversation_id: str, user_id: str, db_path: Path = DB_PATH) -> bool:
    with closing(sqlite3.connect(db_path, timeout=15)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = ? AND user_id = ?",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "INSERT OR IGNORE INTO evicted_conversations(conversation_id, user_id) VALUES (?, ?)",
            (conversation_id, user_id),
        )
        conn.execute("DELETE FROM exchanges WHERE conversation_id = ?", (conversation_id,))
        conn.execute(
            "DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?",
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
            SELECT e.question, e.answer, e.doc_ids, e.status
            FROM exchanges AS e
            JOIN conversations AS c
                ON e.conversation_id = c.conversation_id
            WHERE e.conversation_id = ? AND c.user_id = ?
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

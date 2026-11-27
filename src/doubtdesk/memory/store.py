from time import time_ns
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from doubtdesk.db import TIMESTAMP_FORMAT, connection

MAX_CONVERSATIONS_PER_USER = 30
MAX_CONVERSATION_TITLE_LENGTH = 80


class ConversationLimitReached(ValueError):
    pass


def _title_from_question(question: str) -> str:
    title = " ".join(question.split())
    return title[:MAX_CONVERSATION_TITLE_LENGTH].rstrip() or "New chat"


def save_exchange(
    conversation_id: str,
    question: str,
    answer: str,
    doc_ids: list[str],
    status: str,
    *,
    user_id: str,
    formatted_answer: str | None = None,
    sources: list[dict] | None = None,
    memory_eligible: bool = True,
):
    with connection() as conn:
        owner = conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = %s FOR UPDATE", (conversation_id,),
        ).fetchone()
        if not user_id or owner is None or owner[0] != user_id:
            raise ValueError("Conversation not found.")
        exchange_id = conn.execute(
            """
            INSERT INTO exchanges
                (conversation_id, question, answer, doc_ids, status, formatted_answer, sources, memory_eligible)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                conversation_id,
                question,
                answer,
                Jsonb(doc_ids),
                status,
                formatted_answer,
                Jsonb(sources or []),
                memory_eligible and status not in ("out_of_scope", "verification_failed"),
            ),
        ).fetchone()[0]
        conn.execute(
            """
            UPDATE conversations
            SET last_accessed_ns = %s,
                title = CASE WHEN NOT custom_title AND NOT EXISTS (
                    SELECT 1 FROM exchanges
                    WHERE conversation_id = %s AND id < %s
                ) THEN %s ELSE title END
            WHERE conversation_id = %s
            """,
            (time_ns(), conversation_id, exchange_id, _title_from_question(question), conversation_id),
        )
        return exchange_id


def claim_or_check_conversation(
    conversation_id: str,
    user_id: str,
    max_conversations: int = MAX_CONVERSATIONS_PER_USER,
) -> bool:
    if max_conversations < 1:
        raise ValueError("Maximum conversation count must be positive.")
    with connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"conversations:{user_id}",))
        row = conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
        if row is not None:
            if row[0] != user_id:
                return False
            return True

        if conn.execute(
            "SELECT 1 FROM evicted_conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone():
            return False

        if conn.execute(
            "SELECT 1 FROM exchanges WHERE conversation_id = %s LIMIT 1",
            (conversation_id,),
        ).fetchone():
            return False

        count = conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = %s",
            (user_id,),
        ).fetchone()[0]
        if count >= max_conversations:
            raise ConversationLimitReached(
                f"You have reached the {max_conversations}-conversation limit. Delete a conversation before starting a new one."
            )
        conn.execute(
            """
            INSERT INTO conversations (conversation_id, user_id, last_accessed_ns)
            VALUES (%s, %s, %s)
            ON CONFLICT (conversation_id) DO NOTHING
            """,
            (conversation_id, user_id, time_ns()),
        )
        owner = conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = %s",
            (conversation_id,),
        ).fetchone()
    return owner is not None and owner[0] == user_id


def list_user_conversations(user_id: str) -> list[dict]:
    with connection() as conn:
        return conn.cursor(row_factory=dict_row).execute(
            f"""
            SELECT c.conversation_id, c.title,
                   to_char(c.created_at AT TIME ZONE 'UTC', '{TIMESTAMP_FORMAT}') AS created_at,
                   c.last_accessed_ns,
                   COUNT(e.id) AS message_count
            FROM conversations c
            LEFT JOIN exchanges e ON e.conversation_id = c.conversation_id
            WHERE c.user_id = %s
            GROUP BY c.conversation_id
            ORDER BY c.last_accessed_ns DESC, c.created_at DESC, c.conversation_id DESC
            """,
            (user_id,),
        ).fetchall()


def get_user_conversation(conversation_id: str, user_id: str) -> dict | None:
    with connection() as conn:
        cursor = conn.cursor(row_factory=dict_row)
        row = cursor.execute(
            f"""
            SELECT conversation_id, title,
                   to_char(created_at AT TIME ZONE 'UTC', '{TIMESTAMP_FORMAT}') AS created_at,
                   last_accessed_ns
            FROM conversations WHERE conversation_id = %s AND user_id = %s
            """,
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return None
        exchanges = cursor.execute(
            f"""
            SELECT id, question, answer, COALESCE(formatted_answer, answer) AS formatted_answer,
                   sources, doc_ids, status,
                   to_char(created_at AT TIME ZONE 'UTC', '{TIMESTAMP_FORMAT}') AS created_at
            FROM exchanges WHERE conversation_id = %s ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()
        return {**row, "messages": exchanges}


def rename_user_conversation(conversation_id: str, user_id: str, title: str) -> bool:
    title = " ".join(title.split())
    if not title or len(title) > MAX_CONVERSATION_TITLE_LENGTH:
        raise ValueError(f"Title must be between 1 and {MAX_CONVERSATION_TITLE_LENGTH} characters.")
    with connection() as conn:
        cursor = conn.execute(
            """
            UPDATE conversations SET title = %s, custom_title = TRUE
            WHERE conversation_id = %s AND user_id = %s
            """,
            (title, conversation_id, user_id),
        )
        return cursor.rowcount > 0


def delete_user_conversation(conversation_id: str, user_id: str) -> bool:
    with connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM conversations WHERE conversation_id = %s AND user_id = %s FOR UPDATE",
            (conversation_id, user_id),
        ).fetchone()
        if row is None:
            return False
        conn.execute(
            "INSERT INTO evicted_conversations (conversation_id, user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (conversation_id, user_id),
        )
        conn.execute("DELETE FROM exchanges WHERE conversation_id = %s", (conversation_id,))
        conn.execute(
            "DELETE FROM conversations WHERE conversation_id = %s AND user_id = %s",
            (conversation_id, user_id),
        )
    return True


def get_user_recent_exchanges(
    conversation_id: str,
    user_id: str,
    limit: int = 3,
    *, eligible_only: bool = False,
) -> list[dict]:
    if limit < 1:
        raise ValueError("Limit must be positive.")
    with connection() as conn:
        rows = conn.cursor(row_factory=dict_row).execute(
            """
            SELECT e.id, e.question, e.answer, e.doc_ids, e.status
            FROM exchanges AS e
            JOIN conversations AS c
                ON e.conversation_id = c.conversation_id
            WHERE e.conversation_id = %s AND c.user_id = %s
              AND (NOT %s OR (e.memory_eligible AND e.status NOT IN ('out_of_scope', 'verification_failed')))
            ORDER BY e.id DESC
            LIMIT %s
            """,
            (conversation_id, user_id, eligible_only, limit),
        ).fetchall()
    return list(reversed(rows))


def get_user_eligible_exchanges_by_ids(conversation_id, user_id, ids):
    if not ids:
        return []
    with connection() as conn:
        return conn.cursor(row_factory=dict_row).execute("""SELECT e.id, e.question, e.answer, e.doc_ids, e.status
            FROM exchanges e JOIN conversations c ON c.conversation_id = e.conversation_id
            WHERE e.conversation_id = %s AND c.user_id = %s AND e.memory_eligible
              AND e.status NOT IN ('out_of_scope', 'verification_failed') AND e.id = ANY(%s)
            ORDER BY e.id""", (conversation_id, user_id, list(ids))).fetchall()


def delete_empty_conversation(conversation_id, user_id):
    with connection() as conn:
        cursor = conn.execute(
            """DELETE FROM conversations WHERE conversation_id = %s AND user_id = %s
            AND NOT custom_title AND NOT EXISTS (
                SELECT 1 FROM exchanges WHERE exchanges.conversation_id = conversations.conversation_id
            )""", (conversation_id, user_id))
        return cursor.rowcount == 1

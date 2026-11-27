from doubtdesk.config import runtime_settings
import math
import logging
from doubtdesk.db import connection, to_vector
from doubtdesk.runtime import encode

logger = logging.getLogger(__name__)


def add_exchange_memory(*, exchange_id: int, question: str, answer: str, model):
    text = f"Question: {question}\nAnswer: {answer[:1600]}"
    embedding = encode(model, [text])[0]
    with connection() as conn:
        conn.execute(
            "UPDATE exchanges SET memory_embedding = %s::vector WHERE id = %s",
            (to_vector(embedding), exchange_id),
        )


def search_memory(
    *, conversation_id: str, user_id: str, question: str,
    doc_ids: list[str], limit: int = 3, model,
    exclude_ids=None, max_distance=None,
) -> list[dict]:
    if not conversation_id or not user_id or not question.strip():
        return []
    if limit <= 0:
        return []
    max_distance = runtime_settings().memory_max_distance if max_distance is None else max_distance
    if not 0 <= max_distance <= 2:
        raise ValueError("MEMORY_MAX_DISTANCE must be between 0 and 2")
    excluded = set(exclude_ids or ())
    query_embedding = encode(model, [question])[0]
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT e.id, e.question, e.answer, e.doc_ids, e.memory_embedding <=> %s::vector AS distance
            FROM exchanges e
            JOIN conversations c ON c.conversation_id = e.conversation_id
            WHERE e.conversation_id = %s AND c.user_id = %s AND e.memory_embedding IS NOT NULL
            ORDER BY distance, e.id
            LIMIT 20
            """,
            (to_vector(query_embedding), conversation_id, user_id),
        ).fetchall()
    matched = []
    for exchange_id, prior_question, answer, prior_doc_ids, distance in rows:
        if exchange_id in excluded or not math.isfinite(distance) or distance > max_distance:
            continue
        excluded.add(exchange_id)
        if doc_ids and not set(prior_doc_ids).intersection(doc_ids):
            continue
        matched.append({
            "exchange_id": exchange_id,
            "question": prior_question[:750],
            "answer_excerpt": answer[:1200],
        })
        if len(matched) == limit:
            break
    return matched

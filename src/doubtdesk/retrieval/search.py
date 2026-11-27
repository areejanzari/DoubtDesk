import numpy as np
from doubtdesk.db import connection, to_vector
from doubtdesk.runtime import encode

HNSW_EF_SEARCH = 100
_ITERATIVE_SCAN: bool | None = None


def _supports_iterative_scan(conn) -> bool:
    global _ITERATIVE_SCAN
    if _ITERATIVE_SCAN is None:
        row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        try:
            parts = tuple(int(part) for part in (row[0] if row else "0").split(".")[:2])
        except ValueError:
            parts = (0, 0)
        _ITERATIVE_SCAN = parts >= (0, 8)
    return _ITERATIVE_SCAN


def _configure_vector_search(conn) -> None:
    conn.execute(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}")
    if _supports_iterative_scan(conn):
        conn.execute("SET LOCAL hnsw.iterative_scan = strict_order")

def embed_queries(
    questions: list[str],
    model,
) -> np.ndarray:
    if not questions or any(not question.strip() for question in questions):
        raise ValueError("Questions cannot be empty")

    return encode(model, questions)

def search_chunks(
    question_embedding: np.ndarray,
    doc_ids: list[str],
    top_k: int = 5,
) -> list[list[dict]]:
    if not doc_ids:
        raise ValueError("Atleast one document ID is required.")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results = []
    with connection() as conn:
        _configure_vector_search(conn)
        for embedding in np.atleast_2d(question_embedding):
            rows = conn.execute(
                """
                SELECT chunk_id, doc_id, page_number, text, chunk_index, embedding <=> %s::vector AS distance
                FROM chunks
                WHERE doc_id = ANY(%s)
                ORDER BY distance, chunk_id
                LIMIT %s
                """,
                (to_vector(embedding), list(doc_ids), top_k),
            ).fetchall()
            results.append([
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "page_number": page_number,
                    "text": text,
                    "chunk_index": int(chunk_index),
                    "distance": float(distance),
                }
                for chunk_id, doc_id, page_number, text, chunk_index, distance in rows
            ])

    return results

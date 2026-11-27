import json
import numpy as np
from doubtdesk.db import connection, to_vector
from doubtdesk.ingestion.schemas import ChunkRecord, PageRecord


def replace_document_chunks(
    doc_id: str,
    chunks: list[ChunkRecord],
    embeddings: np.ndarray,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count must match")

    rows = [
        (
            f"{chunk.doc_id}_p{chunk.page_number}_c{chunk.chunk_index}",
            chunk.doc_id,
            chunk.page_number,
            chunk.chunk_index,
            chunk.text,
            to_vector(embedding),
        )
        for chunk, embedding in zip(chunks, embeddings, strict=True)
    ]

    with connection() as conn:
        conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))
        if rows:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO chunks (chunk_id, doc_id, page_number, chunk_index, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s::vector)
                    """,
                    rows,
                )


def replace_document_pages(doc_id: str, pages: list[PageRecord], chunker_version: int | None = None) -> None:
    rows = [
        (
            doc_id, page.page_number, page.text, page.classification, page.ocr_status,
            page.figure_area, json.dumps(page.figure_refs), page.inline_image_count,
            page.has_table, page.watermark_detected, page.visual,
        )
        for page in pages if isinstance(page, PageRecord)
    ]
    with connection() as conn:
        conn.execute("DELETE FROM pages WHERE doc_id = %s", (doc_id,))
        if rows:
            with conn.cursor() as cursor:
                cursor.executemany(
                    """
                    INSERT INTO pages (doc_id, page_number, text, classification, ocr_status, figure_area,
                                       figure_refs, inline_image_count, has_table, watermark_detected, visual)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                    """,
                    rows,
                )
        if chunker_version is not None:
            conn.execute("UPDATE documents SET chunker_version = %s WHERE doc_id = %s", (chunker_version, doc_id))


def page_metadata(pages: list[tuple[str, int]]) -> dict[tuple[str, int], dict]:
    if not pages:
        return {}
    doc_ids = [doc_id for doc_id, _ in pages]
    numbers = [number for _, number in pages]
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT p.doc_id, p.page_number, p.classification, p.ocr_status, p.figure_area, p.figure_refs,
                   p.inline_image_count, p.has_table, p.visual
            FROM pages AS p
            JOIN unnest(%s::text[], %s::int[]) AS wanted(doc_id, page_number)
              ON wanted.doc_id = p.doc_id AND wanted.page_number = p.page_number
            """,
            (doc_ids, numbers),
        ).fetchall()
    return {
        (row[0], int(row[1])): {
            "classification": row[2], "ocr_status": row[3], "figure_area": row[4] or 0.0,
            "figure_refs": row[5] or [], "inline_image_count": row[6] or 0,
            "has_table": bool(row[7]), "visual": bool(row[8]),
        }
        for row in rows
    }


def pages_with_figure_refs(doc_ids: list[str], refs: list[str]) -> list[tuple[str, int]]:
    if not doc_ids or not refs:
        return []
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT doc_id, page_number FROM pages
            WHERE doc_id = ANY(%s) AND figure_refs ?| %s
            ORDER BY doc_id, page_number
            """,
            (list(doc_ids), list(refs)),
        ).fetchall()
    return [(row[0], int(row[1])) for row in rows]


def chunks_for_pages(pages: list[tuple[str, int]], limit_per_page: int = 2) -> list[dict]:
    if not pages:
        return []
    with connection() as conn:
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.doc_id, c.page_number, c.chunk_index, c.text
            FROM chunks AS c
            JOIN unnest(%s::text[], %s::int[]) AS wanted(doc_id, page_number)
              ON wanted.doc_id = c.doc_id AND wanted.page_number = c.page_number
            WHERE c.chunk_index < %s
            ORDER BY c.doc_id, c.page_number, c.chunk_index
            """,
            ([page[0] for page in pages], [page[1] for page in pages], limit_per_page),
        ).fetchall()
    return [{"chunk_id": row[0], "doc_id": row[1], "page_number": int(row[2]), "chunk_index": int(row[3]),
             "text": row[4], "distance": 1.0, "rrf_score": 0.0} for row in rows]


def neighbour_chunks(hits: list[dict], doc_ids: list[str]) -> list[dict]:
    if not hits:
        return []
    keys = [(hit["doc_id"], int(hit["page_number"]), int(hit.get("chunk_index", -1))) for hit in hits]
    if any(key[2] < 0 for key in keys):
        positions = _chunk_positions([hit["chunk_id"] for hit in hits])
        keys = [(hit["doc_id"], int(hit["page_number"]), positions.get(hit["chunk_id"], -1)) for hit in hits]
    with connection() as conn:
        rows = conn.execute(
            """
            WITH hit AS (
                SELECT * FROM unnest(%s::text[], %s::int[], %s::int[]) AS h(doc_id, page_number, chunk_index)
            ),
            last_on_page AS (
                SELECT doc_id, page_number, max(chunk_index) AS last_index
                FROM chunks WHERE doc_id = ANY(%s) GROUP BY doc_id, page_number
            )
            SELECT DISTINCT c.chunk_id, c.doc_id, c.page_number, c.chunk_index, c.text, h.doc_id, h.page_number, h.chunk_index
            FROM hit AS h
            JOIN chunks AS c ON c.doc_id = h.doc_id AND (
                (c.page_number = h.page_number AND c.chunk_index IN (h.chunk_index - 1, h.chunk_index + 1))
                OR (c.page_number = h.page_number + 1 AND c.chunk_index = 0 AND EXISTS (
                    SELECT 1 FROM last_on_page AS l
                    WHERE l.doc_id = h.doc_id AND l.page_number = h.page_number AND l.last_index = h.chunk_index))
                OR (c.page_number = h.page_number - 1 AND h.chunk_index = 0 AND c.chunk_index = (
                    SELECT l.last_index FROM last_on_page AS l
                    WHERE l.doc_id = h.doc_id AND l.page_number = h.page_number - 1))
            )
            WHERE c.doc_id = ANY(%s)
            """,
            ([key[0] for key in keys], [key[1] for key in keys], [key[2] for key in keys], list(doc_ids), list(doc_ids)),
        ).fetchall()
    return [
        {"chunk_id": row[0], "doc_id": row[1], "page_number": int(row[2]), "chunk_index": int(row[3]), "text": row[4],
         "neighbour_of": (row[5], int(row[6]), int(row[7]))}
        for row in rows
    ]


def _chunk_positions(chunk_ids: list[str]) -> dict[str, int]:
    with connection() as conn:
        rows = conn.execute("SELECT chunk_id, chunk_index FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),)).fetchall()
    return {row[0]: int(row[1]) for row in rows}


def document_has_chunks(doc_id: str) -> bool:
    with connection() as conn:
        return conn.execute(
            "SELECT EXISTS (SELECT 1 FROM chunks WHERE doc_id = %s)",
            (doc_id,),
        ).fetchone()[0]

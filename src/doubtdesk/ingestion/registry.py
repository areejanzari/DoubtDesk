import time
from psycopg import Connection
from psycopg.rows import dict_row
from doubtdesk.db import connection


def register_document(
    doc_id: str,
    filename: str,
    sha256: str,
) -> bool:
    with connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO documents
                (doc_id, filename, sha256)
            VALUES (%s, %s, %s)
            ON CONFLICT DO NOTHING
            """,
            (doc_id, filename, sha256),
        )

        return cursor.rowcount == 1


def get_document_by_hash(
    sha256: str,
) -> dict | None:
    with connection() as conn:
        return conn.cursor(row_factory=dict_row).execute(
            """
            SELECT doc_id, filename, sha256, page_count, status
            FROM documents
            WHERE sha256 = %s
            """,
            (sha256,),
        ).fetchone()


def update_document_status(
    doc_id: str,
    status: str,
    page_count: int,
) -> None:
    if status not in {"pending", "processing", "completed", "failed"}:
        raise ValueError(f"Invalid document status: {status}")

    if page_count < 0:
        raise ValueError("Page count cannot be negative")

    with connection() as conn:
        cursor = conn.execute(
            """
            UPDATE documents
            SET status = %s, page_count = %s
            WHERE doc_id = %s
            """,
            (status, page_count, doc_id),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Document not found: {doc_id}")


def is_document_cached(
    sha256: str,
) -> bool:
    document = get_document_by_hash(sha256)

    if document is None:
        return False

    return (
        document["status"] == "completed"
        and document["page_count"] > 0
    )


def get_document_page_counts(
    conn: Connection,
    doc_ids: list[str],
) -> dict[str, int]:
    if not doc_ids:
        return {}

    rows = conn.execute(
        """
        SELECT doc_id, page_count
        FROM documents
        WHERE doc_id = ANY(%s)
          AND status = 'completed'
        """,
        (list(doc_ids),),
    ).fetchall()

    return {
        row[0]: row[1]
        for row in rows
    }


def get_document_names(
    conn: Connection,
    doc_ids: list[str],
    *, user_id: str,
) -> dict[str, str]:
    if not doc_ids:
        return {}

    rows = conn.execute(
        """
        SELECT d.doc_id, ud.filename
        FROM documents AS d
        JOIN user_documents AS ud ON ud.doc_id = d.doc_id
        WHERE ud.user_id = %s AND d.doc_id = ANY(%s)
          AND d.status = 'completed'
        """,
        (user_id, list(doc_ids)),
    ).fetchall()

    return {
        doc_id: filename
        for doc_id, filename in rows
    }


def validate_document_ids(
    conn: Connection,
    doc_ids: list[str],
) -> None:
    if not doc_ids:
        return

    page_counts = get_document_page_counts(conn, doc_ids)

    invalid_ids = [
        doc_id
        for doc_id in doc_ids
        if doc_id not in page_counts
        or page_counts[doc_id] <= 0
    ]

    if invalid_ids:
        raise ValueError(
            f"Invalid or incomplete document IDs: {invalid_ids}"
        )


def list_documents(
    conn: Connection,
    user_id: str,
) -> list[dict]:
    cursor = conn.execute(
        """
        SELECT
            d.doc_id,
            ud.filename,
            d.page_count,
            (SELECT count(*) FROM pages AS p
             WHERE p.doc_id = d.doc_id AND p.ocr_status IN ('failed', 'empty')) AS unreadable
        FROM documents AS d
        JOIN user_documents AS ud
            ON d.doc_id = ud.doc_id
        WHERE ud.user_id = %s
          AND d.status = 'completed'
          AND d.page_count > 0
        ORDER BY ud.filename
        """,
        (user_id,),
    )

    return [
        {
            "doc_id": row[0],
            "filename": row[1],
            "page_count": row[2],
            "unreadable_pages": int(row[3] or 0),
        }
        for row in cursor.fetchall()
    ]


def attach_document_to_user(
    user_id: str,
    doc_id: str,
    filename: str,
    *,
    allow_processing: bool = False,
) -> None:
    with connection() as conn:
        document = conn.execute(
            """
            SELECT doc_id
            FROM documents
            WHERE doc_id = %s
              AND ((status = 'completed' AND page_count > 0)
                   OR (%s AND status = 'processing'))
            """,
            (doc_id, allow_processing),
        ).fetchone()

        if document is None:
            raise ValueError("Document not found or incomplete.")

        conn.execute(
            """
            INSERT INTO user_documents (
                user_id,
                doc_id,
                filename,
                attached_at
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, doc_id)
            DO UPDATE SET filename = excluded.filename
            """,
            (
                user_id,
                doc_id,
                filename,
                int(time.time()),
            ),
        )

def user_can_access_document(
    conn: Connection,
    user_id: str,
    doc_id: str,
    page_number: int,
) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM user_documents AS ud
        JOIN documents AS d
            ON ud.doc_id = d.doc_id
        WHERE ud.user_id = %s
          AND ud.doc_id = %s
          AND d.status = 'completed'
          AND d.page_count >= %s
        """,
        (user_id, doc_id, page_number),
    ).fetchone()

    return row is not None


def validate_user_document_ids(
    conn: Connection,
    user_id: str,
    doc_ids: list[str],
) -> None:
    if not doc_ids:
        return

    unique_ids = list(set(doc_ids))

    rows = conn.execute(
        """
        SELECT d.doc_id
        FROM documents AS d
        JOIN user_documents AS ud
            ON d.doc_id = ud.doc_id
        WHERE ud.user_id = %s
          AND d.doc_id = ANY(%s)
          AND d.status = 'completed'
          AND d.page_count > 0
        """,
        (user_id, unique_ids),
    ).fetchall()

    authorized_ids = {row[0] for row in rows}

    if len(authorized_ids) != len(unique_ids):
        raise ValueError(
            "One or more documents are unavailable."
        )


class DocumentProcessing(ValueError):
    pass


STALE_PROCESSING_SECONDS = 600


def _is_stale(started_at, now):
    return started_at is None or now - started_at > STALE_PROCESSING_SECONDS


def claim_document_ingestion(doc_id, filename, sha256, *, expected_completed=False):
    now = int(time.time())
    with connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"document:{sha256}",))
        row = conn.execute("SELECT doc_id, status, page_count, processing_started_at FROM documents WHERE sha256 = %s", (sha256,)).fetchone()
        if row is None:
            conn.execute("INSERT INTO documents (doc_id, filename, sha256, status, processing_started_at) VALUES (%s, %s, %s, 'processing', %s)", (doc_id, filename, sha256, now))
            return {"doc_id": doc_id, "status": "processing", "page_count": 0}
        existing_id, status, page_count, started_at = row
        if status == "processing" and not _is_stale(started_at, now):
            raise DocumentProcessing("This PDF is still processing. Try again shortly.")
        if status == "completed" and not expected_completed:
            return {"doc_id": existing_id, "status": status, "page_count": page_count}
        conn.execute("UPDATE documents SET status = 'processing', page_count = 0, processing_started_at = %s WHERE doc_id = %s", (now, existing_id))
        return {"doc_id": existing_id, "status": "processing", "page_count": 0}


def touch_document_ingestion(doc_id: str) -> None:
    with connection() as conn:
        conn.execute(
            "UPDATE documents SET processing_started_at = %s WHERE doc_id = %s AND status = 'processing'",
            (int(time.time()), doc_id),
        )


def fail_stale_ingestions() -> int:
    cutoff = int(time.time()) - STALE_PROCESSING_SECONDS
    with connection() as conn:
        return conn.execute(
            """
            UPDATE documents SET status = 'failed', page_count = 0
            WHERE status = 'processing'
              AND (processing_started_at IS NULL OR processing_started_at < %s)
            """,
            (cutoff,),
        ).rowcount


def get_user_document_status(conn: Connection, user_id: str, doc_id: str) -> dict | None:
    row = conn.execute(
        """
        SELECT d.status, d.page_count,
               (SELECT count(*) FROM pages AS p
                WHERE p.doc_id = d.doc_id AND p.ocr_status IN ('failed', 'empty'))
        FROM documents AS d
        JOIN user_documents AS ud ON ud.doc_id = d.doc_id
        WHERE ud.user_id = %s AND d.doc_id = %s
        """,
        (user_id, doc_id),
    ).fetchone()
    if row is None:
        return None
    return {"doc_id": doc_id, "status": row[0], "page_count": row[1], "unreadable_pages": int(row[2] or 0)}

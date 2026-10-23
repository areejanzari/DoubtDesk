import sqlite3
from pathlib import Path


def initialize_registry(
    db_path: Path = Path("data/documents.db"),
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                page_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
            )
            """
        )


def register_document(
    doc_id: str,
    filename: str,
    sha256: str,
    db_path: Path = Path("data/documents.db"),
) -> bool:
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO documents
                (doc_id, filename, sha256)
            VALUES (?, ?, ?)
            """,
            (doc_id, filename, sha256),
        )

        return cursor.rowcount == 1



def get_document_by_hash(
    sha256: str,
    db_path: Path = Path("data/documents.db"),
) -> dict | None:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT doc_id, filename, sha256, page_count, status
            FROM documents
            WHERE sha256 = ?
            """,
            (sha256,),
        ).fetchone()

        return dict(row) if row else None


def update_document_status(
    doc_id: str,
    status: str,
    page_count: int,
    db_path: Path = Path("data/documents.db"),
) -> None:
    if status not in {"pending", "processing", "completed", "failed"}:
        raise ValueError(f"Invalid document status: {status}")

    if page_count < 0:
        raise ValueError("Page count cannot be negative")

    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute(
            """
            UPDATE documents
            SET status = ?, page_count = ?
            WHERE doc_id = ?
            """,
            (status, page_count, doc_id),
        )

        if cursor.rowcount == 0:
            raise ValueError(f"Document not found: {doc_id}")


def is_document_cached(
    sha256: str,
    db_path: Path = Path("data/documents.db"),
) -> bool:
    document = get_document_by_hash(sha256, db_path)

    if document is None:
        return False

    return (
        document["status"] == "completed"
        and document["page_count"] > 0
    )


def get_document_page_counts(
    conn: sqlite3.Connection,
    doc_ids: list[str],
) -> dict[str, int]:
    if not doc_ids:
        return {}

    placeholders = ",".join("?" for _ in doc_ids)

    rows = conn.execute(
        f"""
        SELECT doc_id, page_count
        FROM documents
        WHERE doc_id IN ({placeholders})
          AND status = 'completed'
        """,
        doc_ids,
    ).fetchall()

    return {
        row[0]: row[1]
        for row in rows
    }


def get_document_names(
    conn: sqlite3.Connection,
    doc_ids: list[str],
) -> dict[str, str]:
    if not doc_ids:
        return {}

    placeholders = ", ".join("?" for _ in doc_ids)

    rows = conn.execute(
        f"""
        SELECT doc_id, filename
        FROM documents
        WHERE doc_id IN ({placeholders})
          AND status = 'completed'
        """,
        doc_ids,
    ).fetchall()

    return {
        doc_id: filename
        for doc_id, filename in rows
    }


def validate_document_ids(
    conn: sqlite3.Connection,
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

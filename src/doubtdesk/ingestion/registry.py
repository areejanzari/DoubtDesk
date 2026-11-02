import sqlite3
from pathlib import Path
from contextlib import closing
import time


def initialize_registry(
    db_path: Path = Path("data/documents.db"),
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.execute(
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

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_documents (
                user_id TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                attached_at INTEGER NOT NULL,
                PRIMARY KEY (user_id, doc_id),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_documents_doc
            ON user_documents(doc_id)
            """
        )



def register_document(
    doc_id: str,
    filename: str,
    sha256: str,
    db_path: Path = Path("data/documents.db"),
) -> bool:
    with closing(sqlite3.connect(db_path)) as conn, conn:
        cursor = conn.execute(
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
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute(
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

    with closing(sqlite3.connect(db_path)) as conn, conn:
        cursor = conn.execute(
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


def list_documents(
    conn: sqlite3.Connection,
    user_id: str,
) -> list[dict]:
    cursor = conn.execute(
        """
        SELECT
            d.doc_id,
            ud.filename,
            d.page_count
        FROM documents AS d
        JOIN user_documents AS ud
            ON d.doc_id = ud.doc_id
        WHERE ud.user_id = ?
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
        }
        for row in cursor.fetchall()
    ]


def update_document_filename(
    doc_id: str,
    filename: str,
    db_path: Path = Path("data/documents.db"),
) -> None:
    with closing((sqlite3.connect(db_path))) as conn, conn:
        conn.execute(
            "UPDATE documents SET filename = ? WHERE doc_id = ?",
            (filename, doc_id),
        )


def attach_document_to_user(
    user_id: str,
    doc_id: str,
    filename: str,
    db_path: Path = Path("data/documents.db"),
) -> None:
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")

        document = conn.execute(
            """
            SELECT doc_id
            FROM documents
            WHERE doc_id = ?
              AND status = 'completed'
              AND page_count > 0
            """,
            (doc_id,),
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
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, doc_id)
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
    conn: sqlite3.Connection,
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
        WHERE ud.user_id = ?
          AND ud.doc_id = ?
          AND d.status = 'completed'
          AND d.page_count >= ?
        """,
        (user_id, doc_id, page_number),
    ).fetchone()

    return row is not None


def validate_user_document_ids(
    conn: sqlite3.Connection,
    user_id: str,
    doc_ids: list[str],
) -> None:
    if not doc_ids:
        return

    unique_ids = list(set(doc_ids))
    placeholders = ",".join("?" for _ in unique_ids)

    rows = conn.execute(
        f"""
        SELECT d.doc_id
        FROM documents AS d
        JOIN user_documents AS ud
            ON d.doc_id = ud.doc_id
        WHERE ud.user_id = ?
          AND d.doc_id IN ({placeholders})
          AND d.status = 'completed'
          AND d.page_count > 0
        """,
        [user_id, *unique_ids],
    ).fetchall()

    authorized_ids = {row[0] for row in rows}

    if len(authorized_ids) != len(unique_ids):
        raise ValueError(
            "One or more documents are unavailable."
        )

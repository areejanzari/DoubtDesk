from doubtdesk.config import settings
import re
from threading import Lock
import sqlite3
from contextlib import closing
from pathlib import Path

from doubtdesk.ingestion.schemas import ChunkRecord


LEXICAL_DB = settings.lexical_db
STOPWORDS = frozenset("a an the is are am was were be been being of to in on at for from by with as and or what which who whom how why when where does do did can could would should will may might has have had this that these those it its you your i me my explain describe tell give".split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[.+-][^\W_]+)*", text.casefold(), flags=re.UNICODE)


def initialize_lexical_index(db_path: Path = LEXICAL_DB) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                page_number UNINDEXED,
                text,
                tokenize='unicode61'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS indexed_documents (
                doc_id TEXT PRIMARY KEY
            )
        """)


_initialized_indexes = {}
_index_lock = Lock()


def _ensure_index(db_path):
    path = Path(db_path).resolve()
    with _index_lock:
        if path.exists():
            stat = path.stat()
            if _initialized_indexes.get(path) == (stat.st_dev, stat.st_ino):
                return
        initialize_lexical_index(path)
        stat = path.stat()
        _initialized_indexes[path] = (stat.st_dev, stat.st_ino)


def replace_document_chunks(
    doc_id: str,
    chunks: list[ChunkRecord],
    db_path: Path = LEXICAL_DB,
) -> None:
    _ensure_index(db_path)
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("DELETE FROM chunk_fts WHERE doc_id = ?", (doc_id,))
        conn.executemany(
            "INSERT INTO chunk_fts (chunk_id, doc_id, page_number, text) VALUES (?, ?, ?, ?)",
            [
                (
                    f"{chunk.doc_id}_p{chunk.page_number}_c{chunk.chunk_index}",
                    chunk.doc_id,
                    chunk.page_number,
                    chunk.text,
                )
                for chunk in chunks
            ],
        )
        conn.execute(
            "INSERT OR REPLACE INTO indexed_documents (doc_id) VALUES (?)", (doc_id,)
        )


def lexical_candidates(
    questions: list[str],
    doc_ids: list[str],
    top_k: int = 12,
    db_path: Path = LEXICAL_DB,
) -> list[list[dict]]:
    if not questions or not doc_ids:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    _ensure_index(db_path)
    placeholders = ", ".join("?" for _ in doc_ids)
    statement = f"""
        SELECT chunk_id, doc_id, page_number, text
        FROM chunk_fts
        WHERE chunk_fts MATCH ? AND doc_id IN ({placeholders})
        ORDER BY bm25(chunk_fts), chunk_id
        LIMIT ?
    """
    result = []
    with closing(sqlite3.connect(db_path)) as conn:
        for question in questions:
            words = list(dict.fromkeys(tokenize(question)))
            significant = [word for word in words if word not in STOPWORDS]
            words = significant or words
            if not words:
                result.append([])
                continue
            match_query = " OR ".join('"' + word.replace('"', '""') + '"' for word in words)
            rows = conn.execute(statement, (match_query, *doc_ids, top_k)).fetchall()
            result.append([
                {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "page_number": int(row[2]),
                    "text": row[3],
                }
                for row in rows
            ])
    return result


def backfill_missing_documents(
    collection,
    documents_db: Path = settings.documents_db,
    lexical_db: Path = LEXICAL_DB,
) -> int:
    initialize_lexical_index(lexical_db)
    if not Path(documents_db).is_file():
        return 0
    with closing(sqlite3.connect(documents_db)) as conn:
        doc_ids = [
            row[0] for row in conn.execute(
                "SELECT doc_id FROM documents WHERE status = 'completed'"
            ).fetchall()
        ]
    with closing(sqlite3.connect(lexical_db)) as conn:
        indexed = {
            row[0] for row in conn.execute("SELECT doc_id FROM indexed_documents")
        }
    completed = 0
    for doc_id in doc_ids:
        if doc_id in indexed:
            continue
        chunks = []
        offset = 0
        while True:
            batch = collection.get(
                where={"doc_id": doc_id},
                include=["documents", "metadatas"],
                limit=500,
                offset=offset,
            )
            if not batch["ids"]:
                break
            for document, metadata in zip(
                batch["documents"], batch["metadatas"], strict=True
            ):
                chunks.append(ChunkRecord(
                    doc_id=doc_id,
                    page_number=int(metadata["page_number"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=document,
                ))
            offset += len(batch["ids"])
        replace_document_chunks(doc_id, chunks, lexical_db)
        completed += 1
    return completed

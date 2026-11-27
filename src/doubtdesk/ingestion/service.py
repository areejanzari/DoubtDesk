import logging
from pathlib import Path
from uuid import uuid4
from doubtdesk.ingestion.hashing import calculate_sha256
from doubtdesk.ingestion.registry import (
    claim_document_ingestion,
    DocumentProcessing,
    get_document_by_hash,
    update_document_status,
    attach_document_to_user,
    touch_document_ingestion,
)
from doubtdesk.ingestion.cache import cached_artifacts_exist
from doubtdesk.ingestion.pipeline import ingest_document
from doubtdesk.ingestion.chunking import chunk_document, CHUNKER_VERSION
from doubtdesk.ingestion.embeddings import (
    embed_chunks,
    embedding_tokenizer,
)
from doubtdesk.retrieval.vector_store import replace_document_chunks, replace_document_pages

logger = logging.getLogger(__name__)
HEARTBEAT_PAGES = 10



def build_document(doc_id: str, pdf_path: Path, embedding_model, *, on_page=None, after_extract=None) -> list:
    pages = ingest_document(pdf_path, doc_id, on_page=on_page)
    chunks = chunk_document(pages, embedding_tokenizer(embedding_model))
    if after_extract is not None:
        after_extract()
    embeddings = embed_chunks(chunks, embedding_model)
    replace_document_chunks(doc_id, chunks, embeddings)
    replace_document_pages(doc_id, pages, CHUNKER_VERSION)
    update_document_status(doc_id, "completed", len(pages))
    return pages


def process_pdf(
    pdf_path: Path,
    *,
    user_id: str,
    embedding_model,
) -> dict:
    if not user_id:
        raise ValueError("A user_id is required to upload a document.")

    file_hash = calculate_sha256(pdf_path)

    existing = get_document_by_hash(file_hash)

    if existing and existing["status"] == "processing":
        raise DocumentProcessing("This PDF is still processing. Try again shortly.")
    if existing and existing["status"] == "completed":
        if cached_artifacts_exist(
            existing["doc_id"],
            existing["page_count"],
        ):
            attach_document_to_user(
                user_id=user_id,
                doc_id=existing["doc_id"],
                filename=pdf_path.name,
            )

            return {
                "doc_id": existing["doc_id"],
                "status": "cached",
                "page_count": existing["page_count"],
            }

    claim = claim_document_ingestion(
        str(uuid4()), pdf_path.name, file_hash,
        expected_completed=bool(existing and existing["status"] == "completed"),
    )
    doc_id = claim["doc_id"]
    if claim["status"] == "completed":
        attach_document_to_user(user_id=user_id, doc_id=doc_id, filename=pdf_path.name)
        return {"doc_id": doc_id, "status": "cached", "page_count": claim["page_count"]}

    try:
        pages = build_document(doc_id, pdf_path, embedding_model)

    except Exception:
        update_document_status(doc_id, "failed", 0)
        raise

    attach_document_to_user(
        user_id=user_id,
        doc_id=doc_id,
        filename=pdf_path.name,
    )

    return {
        "doc_id": doc_id,
        "status": "completed",
        "page_count": len(pages),
    }


def start_ingestion(pdf_path: Path, *, filename: str, user_id: str) -> dict:
    if not user_id:
        raise ValueError("A user_id is required to upload a document.")

    file_hash = calculate_sha256(pdf_path)
    existing = get_document_by_hash(file_hash)

    if existing and existing["status"] == "completed" and cached_artifacts_exist(existing["doc_id"], existing["page_count"]):
        attach_document_to_user(user_id=user_id, doc_id=existing["doc_id"], filename=filename)
        return {"doc_id": existing["doc_id"], "status": "cached", "page_count": existing["page_count"], "queued": False}

    try:
        claim = claim_document_ingestion(
            str(uuid4()), filename, file_hash,
            expected_completed=bool(existing and existing["status"] == "completed"),
        )
    except DocumentProcessing:
        current = get_document_by_hash(file_hash)
        if current is None or current["status"] != "processing":
            raise
        attach_document_to_user(user_id=user_id, doc_id=current["doc_id"], filename=filename, allow_processing=True)
        return {"doc_id": current["doc_id"], "status": "processing", "page_count": 0, "queued": False}

    doc_id = claim["doc_id"]
    if claim["status"] == "completed":
        attach_document_to_user(user_id=user_id, doc_id=doc_id, filename=filename)
        return {"doc_id": doc_id, "status": "cached", "page_count": claim["page_count"], "queued": False}

    attach_document_to_user(user_id=user_id, doc_id=doc_id, filename=filename, allow_processing=True)
    return {"doc_id": doc_id, "status": "processing", "page_count": 0, "queued": True}


def run_ingestion(doc_id: str, pdf_path: Path, embedding_model) -> None:
    def heartbeat(page_number):
        if page_number % HEARTBEAT_PAGES == 0:
            touch_document_ingestion(doc_id)

    try:
        build_document(doc_id, pdf_path, embedding_model, on_page=heartbeat,
                       after_extract=lambda: touch_document_ingestion(doc_id))
    except Exception:
        logger.exception("Background ingestion failed for %s", doc_id)
        try:
            update_document_status(doc_id, "failed", 0)
        except Exception:
            logger.exception("Could not mark %s as failed", doc_id)
    finally:
        pdf_path.unlink(missing_ok=True)

from pathlib import Path
from uuid import uuid4
from doubtdesk.ingestion.hashing import calculate_sha256
from doubtdesk.ingestion.registry import (
    initialize_registry,
    register_document,
    get_document_by_hash,
    update_document_status,
    is_document_cached,
)
from doubtdesk.ingestion.cache import cached_artifacts_exist
from doubtdesk.ingestion.pipeline import ingest_document
from doubtdesk.ingestion.chunking import chunk_document
from doubtdesk.ingestion.embeddings import (
    load_embedding_model,
    embed_chunks,
)
from doubtdesk.retrieval.vector_store import (
    get_collection,
    store_chunks,
    delete_document_chunks,
)


def process_pdf(pdf_path: Path) -> dict:
    initialize_registry()

    file_hash = calculate_sha256(pdf_path)
    collection = get_collection()

    existing = get_document_by_hash(file_hash)

    if existing and is_document_cached(file_hash):
        if cached_artifacts_exist(
            existing["doc_id"],
            existing["page_count"],
            collection,
        ):
            return {
                "doc_id": existing["doc_id"],
                "status": "cached",
                "page_count": existing["page_count"],
            }

    doc_id = existing["doc_id"] if existing else str(uuid4())

    if not existing:
        register_document(
            doc_id=doc_id,
            filename=pdf_path.name,
            sha256=file_hash,
        )

    update_document_status(doc_id, "processing", 0)

    try:
        pages = ingest_document(pdf_path, doc_id)
        chunks = chunk_document(pages)

        model = load_embedding_model()
        embeddings = embed_chunks(chunks, model)

        delete_document_chunks(collection, doc_id)

        if chunks:
            store_chunks(collection, chunks, embeddings)

        update_document_status(doc_id, "completed", len(pages))

        return {
            "doc_id": doc_id,
            "status": "completed",
            "page_count": len(pages),
        }

    except Exception:
        update_document_status(doc_id, "failed", 0)
        raise

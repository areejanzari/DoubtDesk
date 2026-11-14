from pathlib import Path
from uuid import uuid4
from doubtdesk.ingestion.hashing import calculate_sha256
from doubtdesk.ingestion.registry import (
    initialize_registry,
    claim_document_ingestion,
    DocumentProcessing,
    get_document_by_hash,
    update_document_status,
    attach_document_to_user,
)
from doubtdesk.ingestion.cache import cached_artifacts_exist
from doubtdesk.ingestion.pipeline import ingest_document
from doubtdesk.ingestion.chunking import chunk_document
from doubtdesk.ingestion.embeddings import (
    embed_chunks,
)
from doubtdesk.retrieval.lexical_store import replace_document_chunks
from doubtdesk.retrieval.vector_store import (
    store_chunks,
    delete_document_chunks,
)



def process_pdf(
    pdf_path: Path,
    *,
    user_id: str,
    embedding_model,
    collection,
) -> dict:
    if not user_id:
        raise ValueError("A user_id is required to upload a document.")

    initialize_registry()

    file_hash = calculate_sha256(pdf_path)

    existing = get_document_by_hash(file_hash)

    if existing and existing["status"] == "processing":
        raise DocumentProcessing("This PDF is still processing. Try again shortly.")
    if existing and existing["status"] == "completed":
        if cached_artifacts_exist(
            existing["doc_id"],
            existing["page_count"],
            collection,
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
        pages = ingest_document(pdf_path, doc_id)
        chunks = chunk_document(pages)

        embeddings = embed_chunks(chunks, embedding_model)

        delete_document_chunks(collection, doc_id)

        if chunks:
            store_chunks(collection, chunks, embeddings)

        replace_document_chunks(doc_id, chunks)

        update_document_status(doc_id, "completed", len(pages))

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

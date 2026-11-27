from doubtdesk.retrieval.vector_store import document_has_chunks
from doubtdesk.storage.images import existing_page_numbers

def cached_artifacts_exist(
    doc_id: str,
    page_count: int,
) -> bool:
    if page_count <= 0:
        return False

    present = existing_page_numbers(doc_id)
    if any(page_number not in present for page_number in range(1, page_count + 1)):
        return False

    return document_has_chunks(doc_id)

from doubtdesk.ingestion.schemas import PageRecord, ChunkRecord

def chunk_text(
        text: str,
        chunk_size: int=1000,
        overlap: int=200,
) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks


def chunk_document(
    pages: list[PageRecord],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[ChunkRecord]:
    all_chunks = []

    for page in pages:
        text_chunks = chunk_text(
            page.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for index, text in enumerate(text_chunks):
            chunk = ChunkRecord(
                doc_id=page.doc_id,
                page_number=page.page_number,
                chunk_index=index,
                text=text,
            )

            all_chunks.append(chunk)

    return all_chunks

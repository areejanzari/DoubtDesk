import argparse
from collections import defaultdict

from doubtdesk.db import connection, initialize_database
from doubtdesk.ingestion.chunking import CHUNKER_VERSION, chunk_document
from doubtdesk.ingestion.embeddings import embed_chunks, embedding_tokenizer, load_embedding_model
from doubtdesk.ingestion.figures import caption_refs, is_visual_page
from doubtdesk.ingestion.layout import clean_document, plain_lines
from doubtdesk.ingestion.schemas import PageRecord
from doubtdesk.retrieval.vector_store import replace_document_chunks, replace_document_pages

LEGACY_OVERLAP = 200


def stitch_legacy_chunks(texts: list[str], overlap: int = LEGACY_OVERLAP) -> str:
    if not texts:
        return ""
    return texts[0] + "".join(text[overlap:] for text in texts[1:])


def outdated_documents(doc_id: str | None) -> list[tuple[str, int]]:
    query = "SELECT doc_id, page_count FROM documents WHERE status = 'completed' AND chunker_version < %s"
    params = [CHUNKER_VERSION]
    if doc_id:
        query += " AND doc_id = %s"
        params.append(doc_id)
    with connection() as conn:
        return [(row[0], int(row[1])) for row in conn.execute(query, params).fetchall()]


def legacy_page_texts(doc_id: str, page_count: int) -> list[str]:
    with connection() as conn:
        rows = conn.execute(
            "SELECT page_number, chunk_index, text FROM chunks WHERE doc_id = %s ORDER BY page_number, chunk_index",
            (doc_id,),
        ).fetchall()
    grouped = defaultdict(list)
    for page_number, _, text in rows:
        grouped[int(page_number)].append(text)
    return [stitch_legacy_chunks(grouped.get(number, [])) for number in range(1, page_count + 1)]


def rebuild(doc_id: str, page_count: int, embedding_model) -> int:
    texts = clean_document([plain_lines(text) for text in legacy_page_texts(doc_id, page_count)])
    pages = []
    for number, text in enumerate(texts, 1):
        refs = caption_refs(text)
        pages.append(PageRecord(
            doc_id=doc_id, page_number=number, text=text, classification="good_text", image_coverage=0.0,
            figure_refs=refs, has_table=any(ref.startswith("table ") for ref in refs),
            visual=is_visual_page(0.0, refs, False),
        ))
    chunks = chunk_document(pages, embedding_tokenizer(embedding_model))
    replace_document_chunks(doc_id, chunks, embed_chunks(chunks, embedding_model))
    replace_document_pages(doc_id, pages, CHUNKER_VERSION)
    return len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Re-chunk documents ingested with an older chunker.")
    parser.add_argument("--doc-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    initialize_database()
    documents = outdated_documents(args.doc_id)
    print(f"{len(documents)} document(s) use an older chunker.")
    if args.dry_run or not documents:
        for doc_id, page_count in documents:
            print(f"  {doc_id} ({page_count} pages)")
        return
    embedding_model = load_embedding_model()
    for doc_id, page_count in documents:
        count = rebuild(doc_id, page_count, embedding_model)
        print(f"  {doc_id}: {count} chunks")
    print("Done. Re-upload a PDF to also restore its figure, watermark and OCR metadata.")


if __name__ == "__main__":
    main()

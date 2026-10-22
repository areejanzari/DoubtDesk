from pathlib import Path
from doubtdesk.ingestion.extractor import extract_document
from doubtdesk.ingestion.pdf_reader import render_pdf_pages


def ingest_document(pdf_path: Path, doc_id: str) -> list[dict]:
    pages = extract_document(pdf_path)

    image_dir = Path("data/page_images") / doc_id
    image_paths = render_pdf_pages(pdf_path, image_dir)

    for page, image_path in zip(pages, image_paths, strict=True):
        page["doc_id"] = doc_id
        page["image_path"] = str(image_path)

    return pages

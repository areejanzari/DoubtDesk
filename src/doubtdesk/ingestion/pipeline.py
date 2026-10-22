from pathlib import Path
from doubtdesk.ingestion.extractor import extract_document
from doubtdesk.ingestion.pdf_reader import render_pdf_pages
from doubtdesk.ingestion.schemas import PageRecord


def ingest_document(
    pdf_path: Path,
    doc_id: str,
) -> list[PageRecord]:
    pages = extract_document(pdf_path)

    image_dir = Path("data/page_images") / doc_id
    image_paths = render_pdf_pages(pdf_path, image_dir)

    records = []

    for page, image_path in zip(pages, image_paths, strict=True):
        record = PageRecord(
            doc_id=doc_id,
            page_number=page["page_number"],
            text=page["text"],
            classification=page["classification"],
            image_coverage=page["image_coverage"],
            image_path=str(image_path),
        )

        records.append(record)

    return records

from doubtdesk.config import settings
from pathlib import Path
import pymupdf
from doubtdesk.ingestion.extractor import extract_page_content
from doubtdesk.ingestion.schemas import PageRecord


def ingest_document(pdf_path: Path, doc_id: str) -> list[PageRecord]:
    image_dir = settings.page_image_dir / doc_id
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with pymupdf.open(pdf_path) as pdf:
        for number, page in enumerate(pdf, 1):
            result = extract_page_content(page)
            image_path = image_dir / f"page_{number:03}.png"
            page.get_pixmap(dpi=150, alpha=False).save(image_path)
            records.append(PageRecord(doc_id=doc_id, page_number=number, image_path=str(image_path), **result))
    return records

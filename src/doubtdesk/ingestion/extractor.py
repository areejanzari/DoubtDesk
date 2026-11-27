import logging
import pymupdf
from pathlib import Path
from doubtdesk.ingestion.quality import classify_page,image_coverage
from doubtdesk.ingestion.ocr import extract_text_with_ocr
from doubtdesk.ingestion.layout import extract_lines, plain_lines

def extract_page_content(page: pymupdf.Page) -> dict:
    text = page.get_text()

    coverage = image_coverage(page)
    classification = classify_page(text,coverage)
    ocr_status = "not_needed"
    lines = None

    if classification == "ocr_candidate":
        try:
            ocr_text = extract_text_with_ocr(page)
            if ocr_text.strip():
                text = ocr_text
                lines = plain_lines(ocr_text)
                ocr_status = "ok"
            else:
                ocr_status = "empty"
        except Exception:
            ocr_status = "failed"
            logging.getLogger(__name__).warning("OCR failed; retaining native page text", exc_info=True)

    if lines is None:
        try:
            lines = extract_lines(page)
        except Exception:
            lines = plain_lines(text)

    return {
        "text": text,
        "classification": classification,
        "image_coverage": coverage,
        "ocr_status": ocr_status,
        "lines": lines,
    }

def extract_document(pdf_path: Path) -> list[dict]:
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            result = extract_page_content(page)

            result["page_number"] = page_index + 1
            pages.append(result)

    return pages

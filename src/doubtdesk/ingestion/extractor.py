import pymupdf
from pathlib import Path
from doubtdesk.ingestion.quality import classify_page,image_coverage
from doubtdesk.ingestion.ocr import extract_text_with_ocr

def extract_page_content(page: pymupdf.Page) -> dict:
    text = page.get_text()

    coverage = image_coverage(page)
    classification = classify_page(text,coverage)

    if classification == "ocr_candidate":
        ocr_text = extract_text_with_ocr(page)

        if ocr_text.strip():
            text = ocr_text

    return {
        "text": text,
        "classification": classification,
        "image_coverage": coverage,
    }

def extract_document(pdf_path: Path) -> list[dict]:
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf):
            result = extract_page_content(page)

            result["page_number"] = page_index + 1
            pages.append(result)

    return pages

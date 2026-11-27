import logging
import pymupdf
from doubtdesk.config import runtime_settings

logger = logging.getLogger(__name__)


def extract_text_with_ocr(
    page: pymupdf.Page,
    language: str | None = None,
    dpi: int | None = None,
) -> str:
    config = runtime_settings()
    text_page = page.get_textpage_ocr(
        language=language or config.ocr_language,
        dpi=dpi or config.ocr_dpi,
        full=True,
    )

    return page.get_text("text", textpage=text_page)


def ocr_available() -> bool:
    try:
        with pymupdf.open() as pdf:
            page = pdf.new_page(width=72, height=72)
            page.get_textpage_ocr(language=runtime_settings().ocr_language, dpi=72, full=True)
        return True
    except Exception as exc:
        logger.warning("OCR is unavailable (%s: %s). Scanned pages will keep their native text only; "
                       "install Tesseract and its language data to enable OCR.", type(exc).__name__, exc)
        return False

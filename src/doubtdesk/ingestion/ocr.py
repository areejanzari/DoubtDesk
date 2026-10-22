import pymupdf

def extract_text_with_ocr(
    page: pymupdf.Page,
    language: str = "eng",
    dpi: int = 300,
) -> str:
    text_page = page.get_textpage_ocr(
        language=language,
        dpi=dpi,
        full=True,
    )

    return page.get_text("text", textpage=text_page)
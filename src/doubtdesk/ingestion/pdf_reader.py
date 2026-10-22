from pathlib import Path
import pymupdf

def extract_page_text(pdf_path: Path) -> list[dict]:
    pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index,page in enumerate(pdf):
            pages.append({
                "page_number": page_index + 1,
                "text": page.get_text(),
            })

    return pages

def render_pdf_pages(pdf_path: Path,
                     output_dir: Path,
                     dpi: int = 150,) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = []

    with pymupdf.open(pdf_path) as pdf:
        for page_index,page in enumerate(pdf):
            pixmap = page.get_pixmap(dpi=dpi, alpha=False)

            image_path = output_dir/ f"page_{page_index + 1:03}.png"
            pixmap.save(image_path)
            image_paths.append(image_path)

    return image_paths
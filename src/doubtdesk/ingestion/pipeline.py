from pathlib import Path
import pymupdf
from doubtdesk.ingestion.extractor import extract_page_content
from doubtdesk.ingestion.figures import layer_names, analyse_document, caption_refs, is_visual_page, page_features
from doubtdesk.ingestion.layout import clean_document
from doubtdesk.ingestion.schemas import PageRecord
from doubtdesk.storage.images import save_page_image


def ingest_document(pdf_path: Path, doc_id: str, *, on_page=None) -> list[PageRecord]:
    contents = []
    features = []
    with pymupdf.open(pdf_path) as pdf:
        layers = layer_names(pdf)
        stats_cache = {}
        for number, page in enumerate(pdf, 1):
            contents.append(extract_page_content(page))
            features.append(page_features(pdf, page, layers, stats_cache))
            image_bytes = page.get_pixmap(dpi=150, alpha=False).tobytes("png")
            save_page_image(doc_id, number, image_bytes)
            if on_page is not None:
                on_page(number)
    texts = clean_document([content["lines"] for content in contents])
    visuals = analyse_document(features)
    records = []
    for number, (content, text, visual) in enumerate(zip(contents, texts, visuals), 1):
        refs = caption_refs(text)
        records.append(PageRecord(
            doc_id=doc_id,
            page_number=number,
            text=text,
            classification=content["classification"],
            image_coverage=visual["figure_area"],
            ocr_status=content["ocr_status"],
            figure_area=visual["figure_area"],
            figure_refs=refs,
            inline_image_count=visual["inline_image_count"],
            has_table=visual["has_table"] or any(ref.startswith("table ") for ref in refs),
            watermark_detected=visual["watermark_detected"],
            visual=is_visual_page(visual["figure_area"], refs, visual["glyph_heavy"]),
        ))
    return records

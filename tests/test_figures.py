from pathlib import Path

import pymupdf
import pytest

from doubtdesk.ingestion.figures import (
    analyse_document, caption_refs, is_visual_page, layer_names, page_features, question_refs,
)
from doubtdesk.ingestion.layout import clean_document, extract_lines
from tests.fixtures.make_pdfs import build

SAMPLES = Path(__file__).resolve().parents[1] / "test_pdfs"


def analyse(path):
    with pymupdf.open(path) as pdf:
        layers = layer_names(pdf)
        features = [page_features(pdf, page, layers) for page in pdf]
        lines = [extract_lines(page) for page in pdf]
    texts = clean_document(lines)
    results = analyse_document(features)
    for result, text in zip(results, texts):
        result["refs"] = caption_refs(text)
        result["visual"] = is_visual_page(result["figure_area"], result["refs"], result["glyph_heavy"])
        result["text"] = text
    return results


@pytest.fixture
def pdf(tmp_path):
    return lambda kind, pages=6: analyse(build(tmp_path / f"{kind}.pdf", kind, pages))


def test_plain_text_pages_are_not_visual(pdf):
    results = pdf("plain")
    assert not any(result["visual"] for result in results)
    assert not any(result["watermark_detected"] for result in results)


def test_transparent_image_watermark_is_background_not_figure(pdf):
    results = pdf("image_watermark")
    assert all(result["watermark_detected"] for result in results)
    assert not any(result["visual"] for result in results)


def test_rotated_text_watermark_is_removed_from_text(pdf):
    results = pdf("text_watermark")
    assert all("REPUBLISHED" not in result["text"] for result in results)
    assert not any(result["visual"] for result in results)


def test_repeated_page_tint_is_template_not_figure(pdf):
    assert not any(result["visual"] for result in pdf("page_tint"))


def test_opaque_full_page_diagram_is_kept_as_figure(pdf):
    results = pdf("full_page_diagram")
    assert results[2]["visual"] and results[2]["figure_area"] > 0.5
    assert not results[2]["watermark_detected"]
    assert not results[0]["visual"]


def test_vector_only_diagram_is_detected(pdf):
    results = pdf("vector_diagram")
    assert results[1]["visual"]
    assert not results[0]["visual"]


def test_captionless_raster_figure_is_detected(pdf):
    results = pdf("captionless_figure")
    assert results[3]["visual"]
    assert results[3]["refs"] == []
    assert not results[0]["visual"]


def test_scanned_book_pages_are_scans_not_watermarks(pdf):
    results = pdf("scan", pages=4)
    assert all(result["scanned"] for result in results)
    assert not any(result["watermark_detected"] for result in results)


def test_running_headers_footers_and_page_numbers_are_removed(pdf):
    results = pdf("plain")
    for result in results:
        assert "BIOLOGY TEXTBOOK CHAPTER" not in result["text"]
        assert "Page " not in result["text"]
        assert "fluid towards the" in result["text"]
    assert "kidney filters blood" in results[0]["text"]


def test_shadow_text_is_deduplicated(pdf):
    text = pdf("shadow_text")[0]["text"]
    assert text.count("ammonotelic") == 1


@pytest.mark.parametrize("text, refs", [
    ("Figure 16.1 Human urinary system", ["figure 16.1"]),
    ("Fig. 10.3 A plot of pressure", ["figure 10.3"]),
    ("Table 10.2: Natural amino acids", ["table 10.2"]),
    ("Scheme 3-4 Mechanism\nFigs 2.1a overview", ["scheme 3.4", "figure 2.1a"]),
    ("as shown in the figure above", []),
])
def test_caption_refs_are_normalised(text, refs):
    assert caption_refs(text) == refs


def test_question_refs_map_diagram_words_to_figures():
    assert question_refs("Explain the diagram 7.3 and Fig 10.1") == ["diagram 7.3", "figure 7.3", "figure 10.1"]
    assert question_refs("What does table 10.4 say?") == ["table 10.4"]
    assert question_refs("What is a nephron?") == []


@pytest.mark.skipif(not (SAMPLES / "kebo116.pdf").exists(), reason="sample textbook PDFs not available")
def test_ncert_watermark_and_figures_on_real_book():
    results = analyse(SAMPLES / "kebo116.pdf")
    assert all(result["watermark_detected"] for result in results)
    assert results[3]["visual"] and "figure 16.4" in results[3]["refs"]
    assert not results[9]["visual"]

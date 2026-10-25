import pymupdf
import pytest
from unittest.mock import patch
from doubtdesk.ingestion.extractor import extract_document, extract_page_content


@pytest.fixture
def sample_pdf(tmp_path):
    pdf_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        for phrase in ["Photosynthesis", "Cell division"]:
            page = pdf.new_page()

            for line in range(25):
                page.insert_text(
                    (50, 50 + line * 20),
                    f"{phrase} is an important biological process.",
                )

        pdf.save(pdf_path)

    return pdf_path


def test_extract_document(sample_pdf):
    result = extract_document(sample_pdf)

    assert len(result) == 2

    assert result[0]["page_number"] == 1
    assert result[1]["page_number"] == 2

    assert "Photosynthesis" in result[0]["text"]
    assert "Cell division" in result[1]["text"]

    assert result[0]["classification"] == "good_text"
    assert result[1]["classification"] == "good_text"

    assert result[0]["image_coverage"] == pytest.approx(0.0)
    

def test_ocr_called_for_scanned_page():
    with pymupdf.open() as pdf:
        page = pdf.new_page()

        with (
            patch(
                "doubtdesk.ingestion.extractor.classify_page",
                return_value="ocr_candidate",
            ),
            patch(
                "doubtdesk.ingestion.extractor.extract_text_with_ocr",
                return_value="Photosynthesis is a biological process.",
            ) as mock_ocr,
        ):
            result = extract_page_content(page)

        mock_ocr.assert_called_once_with(page)
        assert result["text"] == "Photosynthesis is a biological process."
        assert result["classification"] == "ocr_candidate"


def test_ocr_not_called_for_good_text():
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), "Photosynthesis")

        with (
            patch(
                "doubtdesk.ingestion.extractor.classify_page",
                return_value="good_text",
            ),
            patch(
                "doubtdesk.ingestion.extractor.extract_text_with_ocr",
            ) as mock_ocr,
        ):
            result = extract_page_content(page)

        mock_ocr.assert_not_called()
        assert result["classification"] == "good_text"
        assert "Photosynthesis" in result["text"]
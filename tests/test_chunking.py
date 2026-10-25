import pytest
from doubtdesk.ingestion.chunking import chunk_text, chunk_document
from doubtdesk.ingestion.schemas import PageRecord


def test_text_smaller_than_chunk_size():
    result = chunk_text("Photosynthesis", chunk_size=20, overlap=5)

    assert result == ["Photosynthesis"]


def test_text_with_overlap():
    result = chunk_text("ABCDEFGHIJ", chunk_size=6, overlap=2)

    assert result == ["ABCDEF", "EFGHIJ"]


def test_text_without_overlap():
    result = chunk_text("ABCDEFGHIJ", chunk_size=4, overlap=0)

    assert result == ["ABCD", "EFGH", "IJ"]


def test_empty_text():
    assert chunk_text("") == []


def test_whitespace_only_text():
    assert chunk_text("     ", chunk_size=3, overlap=1) == []


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [
        (0, 0),
        (-1, 0),
        (5, 5),
        (5, 6),
        (5, -1),
    ],
)
def test_invalid_chunk_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("Hello world", chunk_size, overlap)


def test_chunk_document_preserves_metadata():
    pages = [
        PageRecord(
            doc_id="doc_001",
            page_number=1,
            text="ABCDEFGHIJ",
            classification="good_text",
            image_coverage=0.0,
            image_path="page_001.png",
        ),
        PageRecord(
            doc_id="doc_001",
            page_number=2,
            text="KLMNOP",
            classification="good_text",
            image_coverage=0.0,
            image_path="page_002.png",
        ),
    ]

    result = chunk_document(pages, chunk_size=6, overlap=2)

    assert len(result) == 3

    assert [chunk.text for chunk in result] == [
        "ABCDEF",
        "EFGHIJ",
        "KLMNOP",
    ]

    assert [chunk.page_number for chunk in result] == [1, 1, 2]
    assert [chunk.chunk_index for chunk in result] == [0, 1, 0]
    assert all(chunk.doc_id == "doc_001" for chunk in result)
import pytest
from unittest.mock import MagicMock
from doubtdesk.answering.validation import (
    validate_evidence_sources,
    validate_evidence_consistency,
)


@pytest.fixture
def pages():
    return [
        {"doc_id": "doc_001", "page_number": 1},
        {"doc_id": "doc_001", "page_number": 2},
        {"doc_id": "doc_002", "page_number": 1},
    ]


def make_evidence(status, answer="", sources=None):
    result = MagicMock()
    result.status = status
    result.answer = answer
    result.sources = sources if sources is not None else []
    return result


def make_source(doc_id, page_numbers):
    source = MagicMock()
    source.doc_id = doc_id
    source.page_numbers = page_numbers
    return source


def test_valid_citations(pages):
    result = make_evidence(
        "sufficient",
        "Photosynthesis converts light energy.",
        [make_source("doc_001", [1, 2])],
    )

    validate_evidence_sources(result, pages)


@pytest.mark.parametrize(
    "doc_id, page_number",
    [
        ("missing_doc", 1),
        ("doc_001", 99),
        ("doc_002", 2),
    ],
)
def test_invalid_citations(pages, doc_id, page_number):
    result = make_evidence(
        "sufficient",
        "Answer",
        [make_source(doc_id, [page_number])],
    )

    with pytest.raises(ValueError, match="Invalid citation"):
        validate_evidence_sources(result, pages)


def test_sufficient_evidence():
    result = make_evidence(
        "sufficient",
        "Photosynthesis produces glucose.",
        [make_source("doc_001", [1])],
    )

    validate_evidence_consistency(result)


def test_sufficient_without_answer():
    result = make_evidence(
        "sufficient",
        "   ",
        [make_source("doc_001", [1])],
    )

    with pytest.raises(ValueError, match="requires an answer"):
        validate_evidence_consistency(result)


def test_sufficient_without_sources():
    result = make_evidence("sufficient", "Answer")

    with pytest.raises(ValueError, match="requires citations"):
        validate_evidence_consistency(result)


def test_sufficient_without_page_numbers():
    result = make_evidence(
        "sufficient",
        "Answer",
        [make_source("doc_001", [])],
    )

    with pytest.raises(ValueError, match="requires page numbers"):
        validate_evidence_consistency(result)


def test_insufficient_evidence():
    result = make_evidence("insufficient")

    validate_evidence_consistency(result)


@pytest.mark.parametrize(
    "answer, has_sources",
    [
        ("Unsupported answer", False),
        ("", True),
        ("Unsupported answer", True),
    ],
)
def test_invalid_insufficient_evidence(answer, has_sources):
    sources = [make_source("doc_001", [1])] if has_sources else []

    result = make_evidence("insufficient", answer, sources)

    with pytest.raises(
        ValueError,
        match="Insufficient evidence must not contain",
    ):
        validate_evidence_consistency(result)
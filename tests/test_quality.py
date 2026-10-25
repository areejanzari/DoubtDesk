import pytest

from doubtdesk.ingestion.quality import (
    count_characters,
    alphabetic_ratio,
    garbage_character_ratio,
    classify_page,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("Hello", 5, id="normal-text"),
        pytest.param("  Hello  ", 5, id="surrounding-spaces"),
        pytest.param("Hello World", 11, id="internal-space"),
        pytest.param("", 0, id="empty-text"),
        pytest.param("   ", 0, id="whitespace-only"),
    ],
)
def test_count_characters(text, expected):
    assert count_characters(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("abc123", 0.5, id="mixed-characters"),
        pytest.param("abc", 1.0, id="all-alphabetic"),
        pytest.param("123", 0.0, id="no-alphabetic"),
        pytest.param("   ", 0.0, id="whitespace-only"),
        pytest.param("a b c", 1.0, id="ignore-whitespace"),
    ],
)
def test_alphabetic_ratio(text, expected):
    assert alphabetic_ratio(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param("abc", 0.0, id="clean-text"),
        pytest.param("\uFFFDabc", 0.25, id="one-garbage-character"),
        pytest.param("\uFFFD\uFFFDabc", 0.4, id="multiple-garbage-characters"),
        pytest.param("", 0.0, id="empty-text"),
        pytest.param("   ", 0.0, id="whitespace-only"),
    ],
)
def test_garbage_character_ratio(text, expected):
    assert garbage_character_ratio(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text, coverage, expected",
    [
        pytest.param("A" * 299, 0.2, "low_text", id="below-good-text-threshold"),
        pytest.param("A" * 300, 0.2, "good_text", id="at-good-text-threshold"),
        pytest.param("A" * 301, 0.2, "good_text", id="above-good-text-threshold"),

        pytest.param("A" * 20, 0.9, "ocr_candidate", id="scanned-page"),
        pytest.param("A" * 20, 0.2, "low_text", id="short-text-page"),

        pytest.param("A" * 49, 0.8, "ocr_candidate", id="ocr-character-boundary"),
        pytest.param("A" * 50, 0.8, "low_text", id="above-ocr-character-boundary"),
        pytest.param("A" * 20, 0.79, "low_text", id="below-image-coverage-threshold"),
        pytest.param("A" * 20, 0.8, "ocr_candidate", id="at-image-coverage-threshold"),

        pytest.param("A" * 90 + "\uFFFD" * 10, 0.2, "low_text", id="exact-garbage-threshold"),
        pytest.param("A" * 89 + "\uFFFD" * 11, 0.2, "ocr_candidate", id="above-garbage-threshold"),
        pytest.param("A" * 400 + "\uFFFD" * 100, 0.2, "ocr_candidate", id="corrupted-long-text"),
    ],
)
def test_classify_page(text, coverage, expected):
    assert classify_page(text, coverage) == expected
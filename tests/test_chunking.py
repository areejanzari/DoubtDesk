import pytest
from doubtdesk.ingestion.chunking import (
    MAX_TOKENS, chunk_document, chunk_page_text, counting_tokenizer, page_units, split_sentences,
)
from doubtdesk.ingestion.embeddings import embedding_tokenizer
from doubtdesk.ingestion.schemas import PageRecord


@pytest.fixture(scope="module")
def tokenizer():
    return embedding_tokenizer()


@pytest.fixture(scope="module")
def counter(tokenizer):
    return counting_tokenizer(tokenizer)


def page(number, text):
    return PageRecord(doc_id="doc_001", page_number=number, text=text, classification="good_text", image_coverage=0.0)


def test_short_text_is_one_chunk(tokenizer):
    assert chunk_page_text("Photosynthesis makes sugar.", tokenizer) == ["Photosynthesis makes sugar."]


def test_empty_and_whitespace_pages_have_no_chunks(tokenizer):
    assert chunk_page_text("", tokenizer) == []
    assert chunk_page_text("   \n\n  ", tokenizer) == []


@pytest.mark.parametrize("target, maximum", [(0, 240), (300, 240), (-1, 10)])
def test_invalid_token_budgets(tokenizer, target, maximum):
    with pytest.raises(ValueError):
        chunk_page_text("Some text.", tokenizer, target_tokens=target, max_tokens=maximum)


def test_long_text_never_exceeds_model_limit(tokenizer, counter):
    sentence = "The glomerular filtration rate of a healthy individual is approximately 125 ml per minute. "
    text = sentence * 80 + " ".join(["word"] * 900)
    chunks = chunk_page_text(text, tokenizer)
    assert len(chunks) > 5
    assert max(len(counter.encode(chunk).ids) for chunk in chunks) <= MAX_TOKENS


def test_chunks_end_on_sentence_boundaries_and_keep_words_whole(tokenizer):
    sentences = [f"Sentence number {index} explains the kidney nephron in detail." for index in range(60)]
    chunks = chunk_page_text(" ".join(sentences), tokenizer)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.endswith("detail.")
        assert chunk.startswith("Sentence number ")


def test_one_sentence_overlap_between_neighbouring_chunks(tokenizer):
    sentences = [f"Fact {index} is about thermal expansion of metals." for index in range(60)]
    first, second = chunk_page_text(" ".join(sentences), tokenizer)[:2]
    last_of_first = split_sentences(first)[-1]
    assert second.startswith(last_of_first)


def test_abbreviations_and_figure_numbers_do_not_split_sentences():
    text = "As shown in Fig. 10.1 the curve rises, e.g. for water. The value is 4.2 J. Next sentence."
    assert split_sentences(text) == [
        "As shown in Fig. 10.1 the curve rises, e.g. for water.",
        "The value is 4.2 J.",
        "Next sentence.",
    ]


def test_headings_start_new_chunks_and_are_carried_as_prefix(tokenizer):
    text = "Intro text about excretion.\n\n16.2\nURINE FORMATION\nUrine formation involves filtration. It also involves secretion."
    chunks = chunk_page_text(text, tokenizer)
    assert chunks[0] == "Intro text about excretion."
    assert chunks[1].startswith("Section: 16.2 URINE FORMATION\n")
    assert "filtration" in chunks[1]


def test_reference_continuation_is_not_mistaken_for_heading():
    text = "The system has two kidneys (Figure\n16.1). Kidneys are bean shaped."
    units = page_units(text)
    assert all(heading is None for _, heading in units)
    assert "(Figure 16.1)." in units[0][0]


def test_line_break_hyphenation_joins_words_but_keeps_numeric_ranges():
    text = "The kidney has an average weight of 120-\n170 g and a re-\nnal pelvis."
    joined = " ".join(unit for unit, _ in page_units(text))
    assert "120- 170 g" in joined
    assert "renal pelvis" in joined


def test_equation_lines_are_kept_intact():
    text = "The relation is derived below.\nαv = 3αl\n(10.9)\nThis completes the proof."
    units = [unit for unit, _ in page_units(text)]
    assert "αv = 3αl\n(10.9)" in units


def test_chunk_document_preserves_metadata(tokenizer):
    pages = [page(1, "Alpha fact one. Alpha fact two."), page(2, "Beta fact.")]
    result = chunk_document(pages, tokenizer)
    assert [chunk.page_number for chunk in result] == [1, 2]
    assert [chunk.chunk_index for chunk in result] == [0, 0]
    assert all(chunk.doc_id == "doc_001" for chunk in result)

from doubtdesk.answering.formatting import format_answer, normalize_answer_citations

DOC_ID = "e8be2d4d-9094-43e8-8cec-7fe4e7853cce"
OTHER_ID = "1c2f3e40-1111-2222-3333-444455556666"


def test_citation_replaced_by_source_index():
    answer = f"Ammonia is excreted. ({DOC_ID}, p. 1)"
    sources = [{"doc_id": DOC_ID, "page_numbers": [1]}]
    assert normalize_answer_citations(answer, sources) == "Ammonia is excreted. [1]"


def test_multiple_document_pages_match_frontend_source_order():
    answer = f"A ({OTHER_ID}, p. 3). B ({DOC_ID}, page 2). C ({DOC_ID}, p. 1)."
    sources = [
        {"doc_id": DOC_ID, "page_numbers": [1, 2]},
        {"doc_id": OTHER_ID, "page_numbers": [3]},
    ]
    assert normalize_answer_citations(answer, sources) == "A [3]. B [2]. C [1]."


def test_uncited_doc_id_does_not_display_in_student_answer():
    answer = f"Some statement ({OTHER_ID}, p. 99)"
    result = normalize_answer_citations(answer, [{"doc_id": DOC_ID, "page_numbers": [1]}])
    assert OTHER_ID not in result
    assert result == "Some statement [unverified source]"


def test_grounded_format_answer_contains_numbered_inline_citation():
    response = {"status": "grounded", "answer": f"Ammonotelic means excreting ammonia. ({DOC_ID}, p. 1)", "sources": [{"doc_id": DOC_ID, "page_numbers": [1]}]}
    formatted = format_answer(response, {DOC_ID: "kebo116.pdf"})
    assert "Ammonotelic means excreting ammonia. [1]" in formatted
    assert "- kebo116.pdf: PDF page(s) 1" in formatted
    assert DOC_ID not in formatted


def test_ordinary_general_answer_is_untouched():
    result = format_answer({"status": "general_knowledge", "answer": "A generic response [1].", "sources": []}, {})
    assert result == "A generic response [1].\n\nNot from document."

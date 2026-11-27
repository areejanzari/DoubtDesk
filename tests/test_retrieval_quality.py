from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pymupdf
import pytest

from doubtdesk.answering.evidence_selection import choose_evidence_route_rules, garbled_ratio
from doubtdesk.db import EMBEDDING_DIMENSIONS
from doubtdesk.evals.metrics import page_metrics, summarize
from doubtdesk.evals.schemas import EvalsRequest
from doubtdesk.ingestion.schemas import ChunkRecord, PageRecord
from doubtdesk.retrieval.hybrid import (
    pin_referenced_pages, reciprocal_rank_fusion, retrieval_queries, with_neighbours,
)
from doubtdesk.retrieval.lexical_store import lexical_candidates, tokenize
from doubtdesk.retrieval.rerank import OnnxCrossEncoder, rerank_chunks
from doubtdesk.retrieval.vector_store import (
    chunks_for_pages, neighbour_chunks, page_metadata, pages_with_figure_refs,
    replace_document_chunks, replace_document_pages,
)


def chunk(doc, page, index, text, **extra):
    return {"chunk_id": f"{doc}_p{page}_c{index}", "doc_id": doc, "page_number": page, "chunk_index": index, "text": text, **extra}


def store(doc, pages):
    records = [ChunkRecord(doc_id=doc, page_number=page, chunk_index=index, text=text)
               for page, texts in pages.items() for index, text in enumerate(texts)]
    replace_document_chunks(doc, records, np.ones((len(records), EMBEDDING_DIMENSIONS)))


def record(doc, number, **fields):
    values = dict(doc_id=doc, page_number=number, text=f"page {number}", classification="good_text", image_coverage=0.0)
    values.update(fields)
    return PageRecord(**values)


def test_retrieval_queries_put_resolved_question_first_and_dedupe():
    assert retrieval_queries("What is GFR?", ["What is GFR?", "Define GFR", " ", "Define GFR"]) == ["What is GFR?", "Define GFR"]


def test_weighted_rank_fusion():
    dense = [{"chunk_id": "a"}, {"chunk_id": "b"}]
    sparse = [{"chunk_id": "b"}, {"chunk_id": "a"}]
    assert [item["chunk_id"] for item in reciprocal_rank_fusion([dense, sparse], weights=[1.0, 0.1])] == ["a", "b"]
    assert [item["chunk_id"] for item in reciprocal_rank_fusion([dense, sparse], weights=[0.1, 1.0])] == ["b", "a"]


@pytest.mark.parametrize("text, tokens", [
    ("Balance H2SO4 + NaOH", ["balance", "h2so4", "naoh"]),
    ("speed in m/s and 10^-3 kg", ["speed", "in", "m/s", "and", "10^-3", "kg"]),
    ("value 4.2 at 25.5 °C", ["value", "4.2", "at", "25.5", "c"]),
])
def test_tokenize_keeps_formulas_and_units_together(text, tokens):
    assert tokenize(text) == tokens


def test_stemmed_and_exact_keyword_search_are_fused():
    store("bio", {1: ["Light reactions happen in the thylakoid."], 2: ["CO2 is absorbed by leaves."]})
    assert lexical_candidates(["Which reaction needs light?"], ["bio"])[0][0]["page_number"] == 1
    assert lexical_candidates(["What absorbs CO2?"], ["bio"])[0][0]["page_number"] == 2
    assert lexical_candidates(["co2"], ["bio"])[0][0]["page_number"] == 2


class FakeSession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask"), SimpleNamespace(name="token_type_ids")]

    def run(self, _, feeds):
        return [feeds["attention_mask"].sum(axis=1, keepdims=True).astype(np.float32)]


def test_cross_encoder_scores_pairs_and_reranks():
    from doubtdesk.ingestion.embeddings import embedding_tokenizer
    from tokenizers import Tokenizer

    encoder = OnnxCrossEncoder.__new__(OnnxCrossEncoder)
    encoder.tokenizer = Tokenizer.from_str(embedding_tokenizer().to_str())
    encoder.tokenizer.no_padding()
    encoder.tokenizer.enable_truncation(max_length=512, strategy="only_second")
    encoder.pad_id = 0
    encoder.session = FakeSession()
    encoder.input_names = {"input_ids", "attention_mask", "token_type_ids"}
    chunks = [chunk("d", 1, 0, "short"), chunk("d", 2, 0, "a much longer passage about kidneys and nephrons")]
    ranked = rerank_chunks(encoder, "kidney", chunks)
    assert [item["page_number"] for item in ranked] == [2, 1]
    assert ranked[0]["rerank_score"] > ranked[1]["rerank_score"]
    assert rerank_chunks(None, "q", chunks) == chunks


def test_cross_encoder_length_sorted_batches_keep_scores_aligned():
    from doubtdesk.ingestion.embeddings import embedding_tokenizer
    from tokenizers import Tokenizer

    encoder = OnnxCrossEncoder.__new__(OnnxCrossEncoder)
    encoder.tokenizer = Tokenizer.from_str(embedding_tokenizer().to_str())
    encoder.tokenizer.no_padding()
    encoder.tokenizer.enable_truncation(max_length=512, strategy="only_second")
    encoder.pad_id = 0
    encoder.session = FakeSession()
    encoder.input_names = {"input_ids", "attention_mask", "token_type_ids"}
    passages = [" ".join(["kidney"] * count) for count in (9, 1, 14, 3, 7, 2, 11, 5, 4, 12, 6)]
    expected = [len(encoder.tokenizer.encode("q", passage).ids) for passage in passages]
    assert encoder.score("q", passages) == [float(value) for value in expected]


def test_pinned_figure_pages_move_first_and_missing_pages_are_fetched():
    ranked = [chunk("d", 1, 0, "one"), chunk("d", 4, 0, "four"), chunk("d", 2, 0, "two")]
    pinned = pin_referenced_pages(ranked, "Explain Figure 16.4 and Fig. 16.6", ["d"],
                                  find_pages=lambda docs, refs: [("d", 4), ("d", 7)],
                                  fetch=lambda pages: [chunk("d", 7, 0, "seven")])
    assert [item["page_number"] for item in pinned] == [4, 7, 1, 2]
    assert all(item.get("pinned_reference") for item in pinned[:2])
    assert pin_referenced_pages(ranked, "What is a nephron?", ["d"], find_pages=lambda *a: 1 / 0, fetch=None) == ranked


def test_neighbours_are_added_as_context_only():
    forwarded = [chunk("d", 3, 2, "hit")]
    found = [chunk("d", 3, 1, "before", neighbour_of=("d", 3, 2)), chunk("d", 4, 0, "next page", neighbour_of=("d", 3, 2))]
    result = with_neighbours(forwarded, ["d"], lookup=lambda hits, docs: found)
    assert [item["chunk_id"] for item in result] == ["d_p3_c2", "d_p3_c1", "d_p4_c0"]
    assert all(item["context_only"] for item in result[1:])
    assert with_neighbours(forwarded, ["d"], lookup=lambda *a: 1 / 0) == forwarded


def test_neighbour_lookup_crosses_page_boundaries():
    store("bio", {1: ["p1 c0", "p1 c1"], 2: ["p2 c0", "p2 c1"], 3: ["p3 c0"]})
    found = {item["chunk_id"] for item in neighbour_chunks([chunk("bio", 1, 1, "p1 c1"), chunk("bio", 3, 0, "p3 c0")], ["bio"])}
    assert found == {"bio_p1_c0", "bio_p2_c0", "bio_p2_c1"}
    assert neighbour_chunks([chunk("bio", 1, 1, "x")], ["other"]) == []


def test_page_metadata_and_figure_ref_lookup():
    store("bio", {1: ["intro"], 4: ["figure page text"]})
    replace_document_pages("bio", [record("bio", 1), record("bio", 4, figure_refs=["figure 16.4"], visual=True, figure_area=0.3)], 2)
    assert pages_with_figure_refs(["bio"], ["figure 16.4", "table 1.1"]) == [("bio", 4)]
    assert pages_with_figure_refs(["other"], ["figure 16.4"]) == []
    meta = page_metadata([("bio", 1), ("bio", 4)])
    assert meta[("bio", 4)]["visual"] and meta[("bio", 4)]["figure_area"] == pytest.approx(0.3)
    assert not meta[("bio", 1)]["visual"]
    assert [item["chunk_id"] for item in chunks_for_pages([("bio", 4)])] == ["bio_p4_c0"]


def rules(question, chunks, meta=None, score=4.0, referenced=None):
    return choose_evidence_route_rules(question, chunks, meta or {}, score, referenced, low_threshold=-5.0)


def test_rules_route_low_relevance_to_low():
    result = rules("Who won the world cup?", [chunk("d", 1, 0, "kidney")], score=-9.0)
    assert result["evidence_mode"] == "low" and result["evidence_reason"] == "low_relevance"
    assert rules("Q", [])["evidence_mode"] == "low"


def test_rules_use_text_for_plain_questions_even_on_figure_pages():
    meta = {("d", 1): {"visual": True, "classification": "good_text"}}
    result = rules("What is GFR?", [chunk("d", 1, 0, "GFR is 125 ml per minute.")], meta)
    assert result["evidence_mode"] == "text"
    assert result["evidence_pages"] == [{"doc_id": "d", "page_number": 1}]


def test_rules_attach_referenced_figure_page_even_if_not_retrieved():
    result = rules("Explain Figure 16.4", [chunk("d", 2, 0, "text")], referenced=[("d", 4)])
    assert result["evidence_mode"] == "visual"
    assert result["retrieved_pages"] == [{"doc_id": "d", "page_number": 4}]
    assert {"doc_id": "d", "page_number": 4} in result["evidence_pages"]


def test_rules_attach_visual_pages_for_visual_questions():
    meta = {("d", 3): {"visual": True}, ("d", 5): {"visual": False}}
    result = rules("Label the parts of the diagram of a nephron", [chunk("d", 5, 0, "a"), chunk("d", 3, 0, "b")], meta)
    assert result["retrieved_pages"] == [{"doc_id": "d", "page_number": 3}]


def test_rules_attach_page_when_extracted_equation_text_is_garbled():
    garbled = "∆ ∆ ∆ V\nV\nl\n= =\n3 3 α\n(10.8)\n→"
    assert garbled_ratio(garbled) > garbled_ratio("Plain sentence about expansion of solids.")
    meta = {("d", 6): {"visual": True}}
    result = rules("Derive the relation between alpha v and alpha l", [chunk("d", 6, 0, garbled)], meta)
    assert result["evidence_mode"] == "visual" and result["evidence_reason"] == "garbled_visual_text"


def test_rules_attach_weak_text_pages_and_skip_context_only_chunks():
    meta = {("d", 2): {"classification": "low_text"}, ("d", 9): {"classification": "ocr_candidate"}}
    result = rules("What happens?", [chunk("d", 2, 0, "x"), chunk("d", 9, 0, "y", context_only=True)], meta)
    assert result["retrieved_pages"] == [{"doc_id": "d", "page_number": 2}]


def test_page_metrics_and_summary():
    ranked = [chunk("a", 3, 0, "x"), chunk("a", 3, 1, "y"), chunk("a", 5, 0, "z")]
    metrics = page_metrics(ranked, {("a", 5)})
    assert metrics["first_relevant_rank"] == 2 and metrics["reciprocal_rank"] == 0.5
    assert metrics["hit@1"] is False and metrics["hit@4"] is True
    assert page_metrics(ranked, set()) is None
    summary = summarize([{"page_metrics": metrics}, {"page_metrics": page_metrics(ranked, {("a", 3)})}, {"page_metrics": None}])
    assert summary == {"questions": 2, "mrr": 0.75, "hit@1": 0.5, "hit@4": 1.0, "hit@8": 1.0, "hit@20": 1.0}


def test_eval_question_caps_depend_on_mode():
    item = {"question": "What is GFR?", "reference_answer": "125 ml/min", "must_include": ["125"],
            "expected_pages": [{"filename": "kebo116.pdf", "page_number": 5}]}
    assert len(EvalsRequest(doc_ids=["d"], questions=[item] * 60, mode="retrieval").questions) == 60
    with pytest.raises(ValueError):
        EvalsRequest(doc_ids=["d"], questions=[item] * 11, mode="full")
    with pytest.raises(ValueError):
        EvalsRequest(doc_ids=["d"], questions=[{**item, "expected_pages": [{"page_number": 1}]}] * 5)


@pytest.mark.parametrize("ocr, status", [
    (lambda page: "Recognised text from the scan.", "ok"),
    (lambda page: "   ", "empty"),
    (lambda page: (_ for _ in ()).throw(RuntimeError("tesseract missing")), "failed"),
])
def test_ocr_status_is_recorded(ocr, status):
    from doubtdesk.ingestion.extractor import extract_page_content
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((50, 50), "A native fact")
        with patch("doubtdesk.ingestion.extractor.classify_page", return_value="ocr_candidate"), \
             patch("doubtdesk.ingestion.extractor.extract_text_with_ocr", side_effect=ocr):
            result = extract_page_content(page)
    assert result["ocr_status"] == status
    assert result["text"].strip()


def test_document_listing_reports_unreadable_pages():
    from doubtdesk.db import connection
    from doubtdesk.ingestion.registry import attach_document_to_user, list_documents, register_document, update_document_status
    register_document("scan_doc", "scan.pdf", "hash_scan")
    update_document_status("scan_doc", "completed", 3)
    attach_document_to_user("user", "scan_doc", "scan.pdf")
    replace_document_pages("scan_doc", [record("scan_doc", 1, ocr_status="failed"), record("scan_doc", 2, ocr_status="empty"),
                                        record("scan_doc", 3)], 2)
    with connection() as conn:
        assert list_documents(conn, "user")[0]["unreadable_pages"] == 2


def test_legacy_chunks_are_stitched_back_into_page_text():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rechunk", Path(__file__).resolve().parents[1] / "scripts" / "rechunk_documents.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = "".join(chr(65 + index % 26) for index in range(2300))
    legacy = [text[start:start + 1000] for start in range(0, len(text), 800)]
    assert module.stitch_legacy_chunks(legacy) == text

from types import SimpleNamespace

import numpy as np

from doubtdesk.db import EMBEDDING_DIMENSIONS
from doubtdesk.ingestion.schemas import ChunkRecord
from doubtdesk.retrieval.lexical_store import lexical_candidates
from doubtdesk.retrieval.vector_store import replace_document_chunks as store_document_chunks
from doubtdesk.answering.evidence_selection import (
    TextEvidenceAssessment,
    EvidencePage,
    choose_evidence_route,
)
from doubtdesk.answering.vision import build_grounded_content


def chunk(doc, page, index, text):
    return ChunkRecord(doc_id=doc, page_number=page, chunk_index=index, text=text)


def replace_document_chunks(doc_id, chunks):
    store_document_chunks(doc_id, chunks, np.ones((len(chunks), EMBEDDING_DIMENSIONS)))


def test_bm25_is_persisted_and_document_scoped():
    replace_document_chunks("biology", [
        chunk("biology", 2, 0, "Photosynthesis needs sunlight and chlorophyll"),
        chunk("biology", 3, 0, "Roots absorb water"),
    ])
    replace_document_chunks("physics", [
        chunk("physics", 1, 0, "Sunlight transfers thermal energy"),
    ])
    results = lexical_candidates(["What does photosynthesis need?"], ["biology"])
    assert results[0][0]["chunk_id"] == "biology_p2_c0"
    assert all(item["doc_id"] == "biology" for item in results[0])
    assert lexical_candidates(["sunlight"], ["physics"])[0][0]["chunk_id"] == "physics_p1_c0"


def test_replacing_document_removes_stale_chunks():
    replace_document_chunks("d", [chunk("d", 1, 0, "old photosynthesis definition")])
    replace_document_chunks("d", [chunk("d", 2, 0, "new respiration definition")])
    assert not lexical_candidates(["photosynthesis"], ["d"])[0]
    assert lexical_candidates(["respiration"], ["d"])[0][0]["page_number"] == 2


def test_rare_terms_outrank_repeated_common_terms():
    replace_document_chunks("bio", [
        chunk("bio", 1, 0, "cycle cycle cycle"),
        chunk("bio", 2, 0, "enzymes of the krebs pathway"),
        chunk("bio", 3, 0, "cycle of water"),
        chunk("bio", 4, 0, "cycle of rocks"),
    ])
    ranked = lexical_candidates(["krebs cycle"], ["bio"])[0]
    assert ranked[0]["chunk_id"] == "bio_p2_c0"
    assert len(ranked) == 4


def test_text_sufficient_uses_no_images():
    chunks = [{"doc_id": "doc", "page_number": 4, "text": "Water boils", "chunk_id": "c1"}]
    assessment = TextEvidenceAssessment(
        sufficiency_score=0.96
    )
    choice = choose_evidence_route(assessment, ["Explain boiling"], chunks, threshold=0.80)
    assert choice["evidence_mode"] == "text"
    assert choice["retrieved_pages"] == []
    content = build_grounded_content("Explain boiling", chunks, [])
    assert not any(item["type"] == "input_image" for item in content)


def test_moderate_score_selects_at_most_two_images_and_ignores_wrong_doc():
    chunks = [
        {"doc_id": "doc", "page_number": i, "text": f"Chunk {i}"}
        for i in range(1, 5)
    ]
    assessment = TextEvidenceAssessment(
        sufficiency_score=0.55,
        recommended_image_pages=[
            EvidencePage(doc_id="other", page_number=1),
            EvidencePage(doc_id="doc", page_number=3),
            EvidencePage(doc_id="doc", page_number=4),
            EvidencePage(doc_id="doc", page_number=2),
        ],
    )
    choice = choose_evidence_route(assessment, ["Explain equation"], chunks, threshold=0.80)
    assert choice["evidence_mode"] == "visual"
    assert choice["retrieved_pages"] == [
        {"doc_id": "doc", "page_number": 3},
        {"doc_id": "doc", "page_number": 4},
    ]


def test_assessment_score_without_chunks_cannot_pass_text_route():
    chunks = [{"doc_id": "doc", "page_number": 1, "text": "Fact"}]
    assessment = TextEvidenceAssessment(
        sufficiency_score=1.0
    )
    choice = choose_evidence_route(assessment, ["Expected requirement"], [], threshold=0.80)
    assert choice["evidence_mode"] == "low"

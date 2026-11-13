from types import SimpleNamespace
import sqlite3

from doubtdesk.ingestion.schemas import ChunkRecord
from doubtdesk.retrieval.lexical_store import (
    replace_document_chunks,
    lexical_candidates,
    backfill_missing_documents,
)
from doubtdesk.answering.evidence_selection import (
    TextEvidenceAssessment,
    ParaphraseEvidence,
    EvidencePage,
    choose_evidence_route,
)
from doubtdesk.answering.vision import build_grounded_content


def chunk(doc, page, index, text):
    return ChunkRecord(doc_id=doc, page_number=page, chunk_index=index, text=text)


def test_bm25_is_persisted_and_document_scoped(tmp_path):
    db = tmp_path / "lexical.db"
    replace_document_chunks("biology", [
        chunk("biology", 2, 0, "Photosynthesis needs sunlight and chlorophyll"),
        chunk("biology", 3, 0, "Roots absorb water"),
    ], db)
    replace_document_chunks("physics", [
        chunk("physics", 1, 0, "Sunlight transfers thermal energy"),
    ], db)
    results = lexical_candidates(["What does photosynthesis need?"], ["biology"], db_path=db)
    assert results[0][0]["chunk_id"] == "biology_p2_c0"
    assert all(item["doc_id"] == "biology" for item in results[0])
    assert lexical_candidates(["sunlight"], ["physics"], db_path=db)[0][0]["chunk_id"] == "physics_p1_c0"


def test_replacing_document_removes_stale_chunks(tmp_path):
    db = tmp_path / "lexical.db"
    replace_document_chunks("d", [chunk("d", 1, 0, "old photosynthesis definition")], db)
    replace_document_chunks("d", [chunk("d", 2, 0, "new respiration definition")], db)
    assert not lexical_candidates(["photosynthesis"], ["d"], db_path=db)[0]
    assert lexical_candidates(["respiration"], ["d"], db_path=db)[0][0]["page_number"] == 2


def test_backfill_existing_document_only_once(tmp_path):
    docs = tmp_path / "documents.db"
    lexical = tmp_path / "lexical.db"
    with sqlite3.connect(docs) as conn:
        conn.execute("CREATE TABLE documents (doc_id TEXT, status TEXT)")
        conn.execute("INSERT INTO documents VALUES ('doc1', 'completed')")

    class Collection:
        calls = 0

        def get(self, **kwargs):
            self.calls += 1
            if kwargs["offset"]:
                return {"ids": [], "documents": [], "metadatas": []}
            return {
                "ids": ["doc1_p1_c0"],
                "documents": ["The heart pumps blood"],
                "metadatas": [{"doc_id": "doc1", "page_number": 1, "chunk_index": 0}],
            }

    collection = Collection()
    assert backfill_missing_documents(collection, docs, lexical) == 1
    assert backfill_missing_documents(collection, docs, lexical) == 0
    assert collection.calls == 2
    assert lexical_candidates(["heart"], ["doc1"], db_path=lexical)[0][0]["chunk_id"] == "doc1_p1_c0"


def test_text_sufficient_uses_no_images():
    chunks = [{"doc_id": "doc", "page_number": 4, "text": "Water boils", "chunk_id": "c1"}]
    assessment = TextEvidenceAssessment(
        scores=[ParaphraseEvidence(paraphrase="Explain boiling", score=0.96, explanation="Clear")]
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
        scores=[ParaphraseEvidence(paraphrase="Explain equation", score=0.55, explanation="Partial")],
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


def test_mismatched_requirements_cannot_pass_text_route():
    chunks = [{"doc_id": "doc", "page_number": 1, "text": "Fact"}]
    assessment = TextEvidenceAssessment(
        scores=[ParaphraseEvidence(paraphrase="Wrong requirement", score=1.0, explanation="Incorrect")]
    )
    choice = choose_evidence_route(assessment, ["Expected requirement"], chunks, threshold=0.80)
    assert choice["evidence_mode"] == "low"

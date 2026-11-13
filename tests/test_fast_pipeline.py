import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from doubtdesk.answering.evidence_selection import (
    ParaphraseEvidence,
    TextEvidenceAssessment,
    assess_text_evidence,
    choose_evidence_route,
)
from doubtdesk.graph.builder import build_graph, build_preparation_graph
from doubtdesk.graph.nodes import evidence_selection_node, retrieve_node
from doubtdesk.retrieval.hybrid import forward_evidence_chunks


PARAPHRASES = [
    "Describe the parts of a nephron in order.",
    "List the nephron components in anatomical sequence.",
    "What is the sequence of nephron structures?",
]


def assessment(values):
    return TextEvidenceAssessment(scores=[
        ParaphraseEvidence(paraphrase=question, score=score, explanation="Test")
        for question, score in zip(PARAPHRASES, values, strict=True)
    ])


def test_assessor_uses_existing_paraphrases_in_one_llm_call():
    client = MagicMock()
    client.responses.parse.return_value.output_parsed = assessment([0.9, 0.8, 1.0])
    chunks = [{"doc_id": "bio", "page_number": 2, "text": "Nephron description"}]
    result = assess_text_evidence(client, "fast-model", "Describe a nephron", PARAPHRASES, chunks)
    assert result.scores[1].paraphrase == PARAPHRASES[1]
    call = client.responses.parse.call_args.kwargs
    assert call["model"] == "fast-model"
    assert call["reasoning"] == {"effort": "none"}
    payload = json.loads(call["input"][1]["content"])
    assert payload["paraphrases"] == PARAPHRASES
    assert payload["retrieved_text"] == chunks
    assert "requirements" not in payload
    assert client.responses.parse.call_count == 1


def test_node_passes_three_paraphrases_directly():
    chunks = [{"doc_id": "bio", "page_number": 1, "text": "nephron"}]
    with patch("doubtdesk.graph.nodes.assess_text_evidence", return_value=assessment([0.8, 0.9, 0.8])) as assessor:
        node = evidence_selection_node(MagicMock(), "fast-model")
        result = node({"question": "Describe nephron", "resolved_question": "Describe nephron", "paraphrases": PARAPHRASES, "retrieved_chunks": chunks})
    assert result["evidence_tier"] == "high"
    assert assessor.call_args.args[3] == PARAPHRASES
    assert "atomic_requirements" not in result


def test_average_route_for_three_paraphrases():
    chunks = [{"doc_id": "bio", "page_number": 1, "text": "nephron"}]
    assert choose_evidence_route(assessment([0.9, 0.8, 0.85]), PARAPHRASES, chunks)["evidence_tier"] == "high"
    assert choose_evidence_route(assessment([0.95, 0.4, 0.5]), PARAPHRASES, chunks)["evidence_tier"] == "moderate"
    assert choose_evidence_route(assessment([0.1, 0.3, 0.4]), PARAPHRASES, chunks)["evidence_tier"] == "low"


def test_wrong_paraphrase_mapping_fails_closed():
    chunks = [{"doc_id": "bio", "page_number": 1, "text": "nephron"}]
    result = assessment([1.0, 1.0, 1.0])
    result.scores[1].paraphrase = "Something else"
    assert choose_evidence_route(result, PARAPHRASES, chunks)["evidence_tier"] == "low"


def test_six_ranked_chunks_forward_four(monkeypatch):
    monkeypatch.delenv("EVIDENCE_FORWARD_TOP_K", raising=False)
    candidates = [{"chunk_id": f"c{i}", "rrf_score": 0.2 - i * 0.01} for i in range(6)]
    assert [chunk["chunk_id"] for chunk in forward_evidence_chunks(candidates)] == ["c0", "c1", "c2", "c3"]


def test_retrieve_node_forwards_four_of_six(tmp_path):
    candidates = [
        {"doc_id": "bio", "chunk_id": f"c{i}", "page_number": i + 1, "text": f"Content{i}", "rrf_score": 0.2 - i * 0.01}
        for i in range(6)
    ]
    with (
        patch("doubtdesk.graph.nodes.validate_user_document_ids"),
        patch("doubtdesk.graph.nodes.hybrid_search", return_value=candidates) as search,
    ):
        node = retrieve_node(MagicMock(), MagicMock(), tmp_path / "documents.db")
        result = node({"question": "Question", "paraphrases": PARAPHRASES, "doc_ids": ["bio"], "user_id": "user"})
    assert result["retrieval_candidate_count"] == 6
    assert len(result["retrieved_chunks"]) == 4
    assert result["retrieved_chunks"] == candidates[:4]
    assert search.call_args.args[2] == PARAPHRASES


def test_graphs_have_no_requirements_stage_and_support_separate_models(tmp_path):
    builder_args = dict(collection=MagicMock(), embedding_model=MagicMock(), client=MagicMock(),
                        model_name="generation-model", paraphrase_model_name="paraphrase-model",
                        assessment_model_name="assessment-model", scope_classifier=MagicMock(),
                        db_path=tmp_path / "documents.db")
    for graph in (build_graph(**builder_args), build_preparation_graph(**builder_args)):
        assert "requirements" not in graph.get_graph().nodes
        assert "select_evidence" in graph.get_graph().nodes
        assert "paraphrase" in graph.get_graph().nodes


def test_preparation_graph_uses_four_chunks_and_assessment_model(tmp_path):
    candidates = [
        {"doc_id": "bio", "chunk_id": f"c{i}", "page_number": i + 1, "text": f"Text{i}", "rrf_score": 0.2 - i * 0.01}
        for i in range(6)
    ]
    assess_models = []
    def fake_assess(client, model_name, question, paraphrases, chunks, *extra):
        assess_models.append((model_name, paraphrases, chunks))
        return assessment([0.9, 0.8, 0.9])
    with (
        patch("doubtdesk.graph.nodes.resolve_and_paraphrase", return_value=SimpleNamespace(resolved_question="Nephron structure", questions=PARAPHRASES)),
        patch("doubtdesk.graph.nodes.validate_user_document_ids"),
        patch("doubtdesk.graph.nodes.hybrid_search", return_value=candidates),
        patch("doubtdesk.graph.nodes.assess_text_evidence", side_effect=fake_assess),
    ):
        classifier = MagicMock()
        classifier.predict.return_value = [{"label": 0} for _ in PARAPHRASES]
        graph = build_preparation_graph(collection=MagicMock(), embedding_model=MagicMock(),
                                        client=MagicMock(), model_name="generation-model",
                                        paraphrase_model_name="prep-model", assessment_model_name="fast-model",
                                        scope_classifier=classifier, db_path=tmp_path / "db")
        result = graph.invoke({"question": "Nephron structure", "doc_ids": ["bio"], "user_id": "u"})
    assert result["evidence_tier"] == "high"
    assert result["retrieval_candidate_count"] == 6
    assert len(result["retrieved_chunks"]) == 4
    assert len(assess_models) == 1
    assert assess_models[0][0] == "fast-model"
    assert assess_models[0][1] == PARAPHRASES
    assert assess_models[0][2] == candidates[:4]

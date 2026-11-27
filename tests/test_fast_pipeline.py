import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doubtdesk.answering.evidence_selection import (
    TextEvidenceAssessment,
    assess_text_evidence,
    choose_evidence_route,
)
from doubtdesk.graph.builder import build_preparation_graph
from doubtdesk.graph.nodes import evidence_selection_node, retrieve_node
from doubtdesk.retrieval.hybrid import forward_evidence_chunks


PARAPHRASES = [
    "Describe the parts of a nephron in order.",
    "List the nephron components in anatomical sequence.",
    "What is the sequence of nephron structures?",
]


def assessment(values):
    return TextEvidenceAssessment(sufficiency_score=sum(values)/len(values))


def test_assessor_uses_existing_paraphrases_in_one_llm_call():
    client = MagicMock()
    client.responses.parse.return_value.output_parsed = assessment([0.9, 0.8, 1.0])
    chunks = [{"doc_id": "bio", "page_number": 2, "text": "Nephron description"}]
    result = assess_text_evidence(client, "fast-model", "Describe a nephron", PARAPHRASES, chunks)
    assert result.sufficiency_score == pytest.approx(0.9)
    call = client.responses.parse.call_args.kwargs
    assert call["model"] == "fast-model"
    assert call["reasoning"] == {"effort": "none"}
    payload = json.loads(call["input"][1]["content"])
    assert payload["paraphrases"] == PARAPHRASES
    assert payload["retrieved_text"] == chunks
    assert "requirements" not in payload
    assert client.responses.parse.call_count == 1


def test_llm_assessment_node_is_available_behind_flag(monkeypatch):
    monkeypatch.setenv("EVIDENCE_ASSESSMENT_ENABLED", "true")
    chunks = [{"doc_id": "bio", "page_number": 1, "text": "nephron"}]
    with patch("doubtdesk.graph.nodes.aassess_text_evidence", new_callable=AsyncMock,
               return_value=assessment([0.8, 0.9, 0.8])) as assessor:
        node = evidence_selection_node(MagicMock(), "fast-model")
        result = node({"question": "Describe nephron", "resolved_question": "Describe nephron", "paraphrases": PARAPHRASES, "retrieved_chunks": chunks})
    assert result["evidence_tier"] == "high"
    assert assessor.call_args.args[3] == PARAPHRASES
    assert "atomic_requirements" not in result


def test_rule_node_makes_no_llm_call_by_default(monkeypatch):
    monkeypatch.delenv("EVIDENCE_ASSESSMENT_ENABLED", raising=False)
    chunks = [{"doc_id": "bio", "page_number": 1, "text": "nephron", "rerank_score": 4.0}]
    client = MagicMock()
    with patch("doubtdesk.graph.nodes.aassess_text_evidence", new_callable=AsyncMock) as assessor:
        node = evidence_selection_node(client, "fast-model")
        result = node({"question": "Describe nephron", "paraphrases": PARAPHRASES, "retrieved_chunks": chunks,
                       "retrieval_top_score": 4.0, "doc_ids": ["bio"]})
    assessor.assert_not_called()
    client.responses.parse.assert_not_called()
    assert result["evidence_mode"] == "text"
    assert result["evidence_pages"] == [{"doc_id": "bio", "page_number": 1}]


def test_average_route_for_three_paraphrases():
    chunks = [{"doc_id": "bio", "page_number": 1, "text": "nephron"}]
    assert choose_evidence_route(assessment([0.9, 0.8, 0.85]), PARAPHRASES, chunks)["evidence_tier"] == "high"
    assert choose_evidence_route(assessment([0.95, 0.4, 0.5]), PARAPHRASES, chunks)["evidence_tier"] == "moderate"
    assert choose_evidence_route(assessment([0.1, 0.3, 0.4]), PARAPHRASES, chunks)["evidence_tier"] == "low"


def test_paraphrase_rewording_does_not_change_route():
    chunks = [{"doc_id": "bio", "page_number": 1, "text": "nephron"}]
    result = assessment([1.0, 1.0, 1.0])
    assert choose_evidence_route(result, ["Something else", "‘Question?’", "Question."], chunks)["evidence_tier"] == "high"


def test_ranked_chunks_forward_configured_count(monkeypatch):
    monkeypatch.delenv("EVIDENCE_FORWARD_TOP_K", raising=False)
    candidates = [{"chunk_id": f"c{i}", "doc_id": "bio", "page_number": i, "text": f"distinct text {i}"} for i in range(8)]
    assert [chunk["chunk_id"] for chunk in forward_evidence_chunks(candidates)] == [f"c{i}" for i in range(6)]


def test_near_duplicate_chunks_from_same_page_are_dropped():
    text = "The nephron is the functional unit of the kidney with glomerulus and tubule."
    candidates = [
        {"chunk_id": "a", "doc_id": "bio", "page_number": 1, "text": text},
        {"chunk_id": "b", "doc_id": "bio", "page_number": 1, "text": text + " Indeed."},
        {"chunk_id": "c", "doc_id": "bio", "page_number": 2, "text": text},
    ]
    assert [chunk["chunk_id"] for chunk in forward_evidence_chunks(candidates)] == ["a", "c"]


def test_retrieve_node_searches_with_resolved_question_and_paraphrases(tmp_path):
    candidates = [
        {"doc_id": "bio", "chunk_id": f"c{i}", "page_number": i + 1, "text": f"Content{i}", "rerank_score": 5.0 - i}
        for i in range(8)
    ]
    with (
        patch("doubtdesk.graph.nodes.validate_user_document_ids"),
        patch("doubtdesk.graph.nodes.aretrieve_ranked_evidence", new_callable=AsyncMock,
              return_value=(candidates, candidates[:6])) as search,
    ):
        node = retrieve_node(MagicMock(), reranker="reranker")
        result = node({"question": "Question", "resolved_question": "Resolved question", "paraphrases": PARAPHRASES,
                       "doc_ids": ["bio"], "user_id": "user"})
    assert result["retrieval_candidate_count"] == 8
    assert result["retrieved_chunks"] == candidates[:6]
    assert result["retrieval_top_score"] == 5.0
    assert search.call_args.args[1] == ["Resolved question", *PARAPHRASES]
    assert search.call_args.kwargs["question"] == "Resolved question"
    assert search.call_args.kwargs["reranker"] == "reranker"


def test_graph_has_no_requirements_stage_and_supports_separate_models(tmp_path):
    graph = build_preparation_graph(embedding_model=MagicMock(), client=MagicMock(),
                                    model_name="generation-model", paraphrase_model_name="paraphrase-model",
                                    assessment_model_name="assessment-model", scope_classifier=MagicMock())
    assert "requirements" not in graph.get_graph().nodes
    assert "select_evidence" in graph.get_graph().nodes
    assert "paraphrase" in graph.get_graph().nodes


def test_preparation_graph_uses_assessment_model_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("EVIDENCE_ASSESSMENT_ENABLED", "true")
    candidates = [
        {"doc_id": "bio", "chunk_id": f"c{i}", "page_number": i + 1, "text": f"Text{i}", "rrf_score": 0.2 - i * 0.01}
        for i in range(6)
    ]
    assess_models = []

    async def fake_assess(client, model_name, question, paraphrases, chunks, *extra):
        assess_models.append((model_name, paraphrases, chunks))
        return assessment([0.9, 0.8, 0.9])

    with (
        patch("doubtdesk.graph.nodes.aresolve_and_paraphrase", new_callable=AsyncMock,
              return_value=SimpleNamespace(resolved_question="Nephron structure", questions=PARAPHRASES)),
        patch("doubtdesk.graph.nodes.validate_user_document_ids"),
        patch("doubtdesk.graph.nodes.aretrieve_ranked_evidence", new_callable=AsyncMock,
              return_value=(candidates, candidates[:4])),
        patch("doubtdesk.graph.nodes.aassess_text_evidence", side_effect=fake_assess),
    ):
        classifier = MagicMock()
        classifier.predict.return_value = [{"label": 0} for _ in PARAPHRASES]
        graph = build_preparation_graph(embedding_model=MagicMock(),
                                        client=MagicMock(), model_name="generation-model",
                                        paraphrase_model_name="prep-model", assessment_model_name="fast-model",
                                        scope_classifier=classifier)
        result = graph.invoke({"question": "Nephron structure", "doc_ids": ["bio"], "user_id": "u"})
    assert result["evidence_tier"] == "high"
    assert result["retrieval_candidate_count"] == 6
    assert len(result["retrieved_chunks"]) == 4
    assert len(assess_models) == 1
    assert assess_models[0][0] == "fast-model"
    assert assess_models[0][1] == PARAPHRASES
    assert assess_models[0][2] == candidates[:4]

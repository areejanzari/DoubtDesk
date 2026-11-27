import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doubtdesk.api.streaming import PreparedAnswerGraph
from doubtdesk.graph.builder import build_preparation_graph
from doubtdesk.graph.nodes import retrieve_node
from doubtdesk.graph.routing import route_documents, route_retrieval


@pytest.fixture(autouse=True)
def mock_paraphrasing(monkeypatch):
    async def fake_paraphrasing(**kwargs):
        question = kwargs["question"]
        return SimpleNamespace(resolved_question=question, questions=[question, question, question])

    monkeypatch.setattr("doubtdesk.graph.nodes.aresolve_and_paraphrase", fake_paraphrasing)


class FakeStream:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        for start in range(0, len(self.text), 7):
            yield SimpleNamespace(type="response.output_text.delta", delta=self.text[start:start + 7])

    def get_final_response(self):
        return SimpleNamespace(status="completed", usage=None)


def provider(text):
    client = MagicMock()
    client.responses.stream.side_effect = lambda **kwargs: FakeStream(text)
    return client


def answering_graph(client, judge=None):
    classifier = MagicMock()
    classifier.predict.return_value = [{"label": 0}] * 3
    preparation = build_preparation_graph(embedding_model=MagicMock(), client=client,
                                          model_name="test-model", scope_classifier=classifier)
    return PreparedAnswerGraph(preparation, client, "test-model", judge)


def test_route_documents_with_pdf():
    assert route_documents({"question": "What is photosynthesis?", "doc_ids": ["doc_001"]}) == "retrieve"


def test_route_documents_without_pdf():
    assert route_documents({"question": "What is photosynthesis?", "doc_ids": []}) == "fallback"


def test_retrieve_node_empty_question(tmp_path):
    retrieve = retrieve_node(embedding_model=MagicMock())
    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve({"question": "   ", "doc_ids": ["doc_001"]})


def test_retrieve_node_without_documents(tmp_path):
    retrieve = retrieve_node(embedding_model=MagicMock())
    assert retrieve({"question": "Explain gravity", "doc_ids": []}) == {"retrieved_pages": [], "retrieved_chunks": []}


def test_graph_without_documents(tmp_path):
    graph = answering_graph(provider("Gravity is a force of attraction."))
    result = graph.invoke({"question": "What is gravity?", "doc_ids": [], "user_id": "user_A"})
    assert result["status"] == "general_knowledge"
    assert result["answer"] == "Gravity is a force of attraction."
    assert result["sources"] == []
    assert "Not from document." in result["formatted_answer"]


def test_graph_grounded_route_uses_text_when_rules_find_no_visual_need(tmp_path):
    raw = json.dumps({"status": "sufficient", "answer": "Photosynthesis converts light energy (S1).", "sources": ["S1"]})
    chunks = [{"doc_id": "doc_001", "chunk_id": "a", "chunk_index": 0, "page_number": 1,
               "text": "Photosynthesis converts light energy", "distance": 0.1, "rrf_score": 0.03, "rerank_score": 6.0}]
    with (
        patch("doubtdesk.graph.nodes.validate_user_document_ids") as owner,
        patch("doubtdesk.graph.nodes.aretrieve_ranked_evidence", new_callable=AsyncMock, return_value=(chunks, chunks)),
        patch("doubtdesk.graph.nodes.get_page_image_paths") as images,
        patch("doubtdesk.api.streaming.get_document_names", return_value={"doc_001": "biology.pdf"}),
    ):
        graph = answering_graph(provider(raw))
        result = graph.invoke({"question": "What is photosynthesis?", "doc_ids": ["doc_001"], "user_id": "user_A"})
    owner.assert_called_once()
    images.assert_not_called()
    assert result["status"] == "grounded"
    assert result["answer"] == "Photosynthesis converts light energy [1]."
    assert result["evidence_mode"] == "text"
    assert "biology.pdf" in result["formatted_answer"]


def test_graph_low_relevance_routes_document_question_to_not_in_pdf(tmp_path):
    chunks = [{"doc_id": "doc_001", "chunk_id": "a", "chunk_index": 0, "page_number": 1,
               "text": "Unrelated text", "rerank_score": -9.0}]
    client = provider("unused")
    with (
        patch("doubtdesk.graph.nodes.validate_user_document_ids"),
        patch("doubtdesk.graph.nodes.aretrieve_ranked_evidence", new_callable=AsyncMock, return_value=(chunks, chunks)),
    ):
        result = answering_graph(client).invoke({
            "question": "According to this PDF, who won the 2018 World Cup?", "doc_ids": ["doc_001"], "user_id": "user_A",
        })
    assert result["evidence_mode"] == "low"
    assert result["document_insufficient"] is True
    client.responses.stream.assert_not_called()


def test_graph_judge_is_called_once_for_approved_answer(tmp_path):
    from doubtdesk.answering.output_guardrails import JudgeVerdict

    judge = MagicMock()
    judge.evaluate.return_value = JudgeVerdict(passed=True, relevance=1, groundedness=1, completeness=1, safety="pass")
    result = answering_graph(provider("Gravity attracts masses"), judge).invoke(
        {"question": "What is gravity?", "doc_ids": [], "user_id": "u"})
    assert result["judge_passed"] is True
    assert result["answer"] == "Gravity attracts masses"
    assert judge.evaluate.call_count == 1


def test_route_retrieval_with_chunks():
    assert route_retrieval({"retrieved_chunks": [{"doc_id": "doc_001", "page_number": 1, "text": "Some text"}]}) == "evidence"


def test_route_retrieval_without_chunks():
    assert route_retrieval({"retrieved_chunks": []}) == "fallback"

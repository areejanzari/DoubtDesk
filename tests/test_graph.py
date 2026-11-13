from unittest.mock import MagicMock, patch
import pytest
from doubtdesk.graph.routing import route_documents, route_evidence, route_retrieval
from doubtdesk.graph.nodes import (
    retrieve_node,
    evidence_node,
    fallback_node,
    grounded_node,
    formatting_node,
)
from doubtdesk.graph.builder import build_graph
from types import SimpleNamespace

@pytest.fixture(autouse=True)
def mock_paraphrasing(monkeypatch):
    def fake_paraphrasing(**kwargs):
        question = kwargs["question"]

        return SimpleNamespace(
            resolved_question=question,
            questions=[question, question, question],
        )

    monkeypatch.setattr(
        "doubtdesk.graph.nodes.resolve_and_paraphrase",
        fake_paraphrasing,
    )

def test_route_documents_with_pdf():
    state = {"question": "What is photosynthesis?", "doc_ids": ["doc_001"]}

    assert route_documents(state) == "retrieve"


def test_route_documents_without_pdf():
    state = {"question": "What is photosynthesis?", "doc_ids": []}

    assert route_documents(state) == "fallback"


@pytest.mark.parametrize(
    "status, expected",
    [
        ("sufficient", "grounded"),
        ("insufficient", "fallback"),
    ],
)
def test_route_evidence(status, expected):
    evidence = MagicMock()
    evidence.status = status

    assert route_evidence({"evidence": evidence}) == expected

def test_retrieve_node_empty_question(tmp_path):
    retrieve = retrieve_node(
        collection=MagicMock(),
        embedding_model=MagicMock(),
        db_path=tmp_path / "test.db",
    )

    with pytest.raises(ValueError, match="Question cannot be empty"):
        retrieve({"question": "   ", "doc_ids": ["doc_001"]})


def test_retrieve_node_without_documents(tmp_path):
    retrieve = retrieve_node(
        collection=MagicMock(),
        embedding_model=MagicMock(),
        db_path=tmp_path / "test.db",
    )

    result = retrieve({
        "question": "Explain gravity",
        "doc_ids": [],
    })

    assert result == {"retrieved_pages": [], "retrieved_chunks": []}

def test_evidence_node_without_pages():
    evaluate = evidence_node(MagicMock(), "test-model")

    with pytest.raises(
        ValueError,
        match="Evidence evaluation requires retrieved text or pages",
    ):
        evaluate({
            "question": "Explain gravity",
            "retrieved_pages": [],
        })


def test_evidence_node_with_text_only():
    evidence = MagicMock()
    evidence.status = "sufficient"
    pages = [{"doc_id": "doc_001", "page_number": 1}]
    chunks = [{"doc_id": "doc_001", "page_number": 1, "text": "Photosynthesis uses light"}]

    with (
        patch("doubtdesk.graph.nodes.build_grounded_content", return_value=["text-content"]) as content,
        patch("doubtdesk.graph.nodes.generate_vision_answer", return_value=evidence),
        patch("doubtdesk.graph.nodes.validate_evidence_consistency") as validate_consistency,
        patch("doubtdesk.graph.nodes.validate_evidence_sources") as validate_sources,
    ):
        evaluate = evidence_node(MagicMock(), "test-model")
        result = evaluate({
            "question": "What is photosynthesis?",
            "retrieved_chunks": chunks,
            "retrieved_pages": [],
            "evidence_pages": pages,
        })

    assert result["evidence"] is evidence
    content.assert_called_once_with(question="What is photosynthesis?", chunks=chunks, pages=[])
    validate_consistency.assert_called_once_with(evidence)
    validate_sources.assert_called_once_with(evidence, pages)


def test_fallback_node():
    with patch(
        "doubtdesk.graph.nodes.generate_general_answer",
        return_value="Gravity attracts objects.",
    ):
        fallback = fallback_node(MagicMock(), "test-model")

        result = fallback({"question": "Explain gravity"})

    assert result == {
        "status": "general_knowledge",
        "answer": "Gravity attracts objects.",
        "sources": [],
    }


def test_grounded_node():
    source = MagicMock()
    source.model_dump.return_value = {
        "doc_id": "doc_001",
        "page_numbers": [2],
    }

    evidence = MagicMock()
    evidence.status = "sufficient"
    evidence.answer = "Photosynthesis converts light energy."
    evidence.sources = [source]

    result = grounded_node({"evidence": evidence})

    assert result == {
        "status": "grounded",
        "answer": "Photosynthesis converts light energy.",
        "sources": [
            {"doc_id": "doc_001", "page_numbers": [2]},
        ],
    }


def test_grounded_node_rejects_insufficient_evidence():
    evidence = MagicMock()
    evidence.status = "insufficient"

    with pytest.raises(
        ValueError,
        match="Grounded answer requires sufficient evidence",
    ):
        grounded_node({"evidence": evidence})

def test_formatting_node(tmp_path):
    db_path = tmp_path / "test.db"

    with (
        patch(
            "doubtdesk.graph.nodes.get_document_names",
            return_value={"doc_001": "biology.pdf"},
        ),
        patch(
            "doubtdesk.graph.nodes.format_answer",
            return_value="Formatted answer",
        ) as mock_format,
    ):
        formatter = formatting_node(db_path)

        result = formatter({
            "status": "grounded",
            "answer": "Photosynthesis uses sunlight.",
            "sources": [
                {"doc_id": "doc_001", "page_numbers": [1]},
            ],
        })

    assert result == {"formatted_answer": "Formatted answer"}
    mock_format.assert_called_once()

def test_graph_without_documents(tmp_path):
    db_path = tmp_path / "test.db"

    with patch(
        "doubtdesk.graph.nodes.generate_general_answer",
        return_value="Gravity is a force of attraction.",
    ):

        classifier = MagicMock()
        classifier.predict.return_value = [
            {"label": 0},
            {"label": 0},
            {"label": 0},
        ]
        graph = build_graph(
            scope_classifier=classifier,
            collection=MagicMock(),
            embedding_model=MagicMock(),
            client=MagicMock(),
            model_name="test-model",
            db_path=db_path,
        )

        result = graph.invoke({
            "question": "What is gravity?",
            "doc_ids": [],
            "user_id": "user_A",
        })

    assert result["status"] == "general_knowledge"
    assert result["answer"] == "Gravity is a force of attraction."
    assert result["sources"] == []
    assert "Not from document." in result["formatted_answer"]


def test_graph_grounded_route_uses_text_when_sufficient(tmp_path):
    evidence = MagicMock()
    evidence.status = "sufficient"
    evidence.answer = "Photosynthesis converts light energy."
    source = MagicMock()
    source.model_dump.return_value = {"doc_id": "doc_001", "page_numbers": [1]}
    evidence.sources = [source]
    chunks = [{"doc_id": "doc_001", "chunk_id": "a", "page_number": 1,
               "text": "Photosynthesis converts light energy", "distance": 0.1, "rrf_score": 0.03}]
    assessment = SimpleNamespace(scores=[SimpleNamespace(paraphrase="What is photosynthesis?", score=0.95)] * 3)

    with (
        patch("doubtdesk.graph.nodes.validate_user_document_ids") as owner,
        patch("doubtdesk.graph.nodes.hybrid_search", return_value=chunks),
        patch("doubtdesk.graph.nodes.assess_text_evidence", return_value=assessment),
        patch("doubtdesk.graph.nodes.get_page_image_paths") as images,
        patch("doubtdesk.graph.nodes.generate_vision_answer", return_value=evidence),
        patch("doubtdesk.graph.nodes.validate_evidence_consistency"),
        patch("doubtdesk.graph.nodes.validate_evidence_sources"),
        patch("doubtdesk.graph.nodes.get_document_names", return_value={"doc_001": "biology.pdf"}),
    ):
        classifier = MagicMock()
        classifier.predict.return_value = [{"label": 0}] * 3
        graph = build_graph(
            scope_classifier=classifier,
            collection=MagicMock(),
            embedding_model=MagicMock(),
            client=MagicMock(),
            model_name="test-model",
            db_path=tmp_path / "test.db",
        )
        result = graph.invoke({
            "question": "What is photosynthesis?",
            "doc_ids": ["doc_001"],
            "user_id": "user_A",
        })

    owner.assert_called_once()
    images.assert_not_called()
    assert result["status"] == "grounded"
    assert result["answer"] == evidence.answer
    assert result["evidence_mode"] == "text"
    assert "biology.pdf" in result["formatted_answer"]


def test_route_retrieval_with_chunks():
    state = {
        "retrieved_chunks": [
            {"doc_id": "doc_001", "page_number": 1, "text": "Some text"}
        ]
    }

    assert route_retrieval(state) == "evidence"


def test_route_retrieval_without_chunks():
    state = {"retrieved_chunks": []}

    assert route_retrieval(state) == "fallback"
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from doubtdesk.answering.service import answer_question, resolve_answer
from doubtdesk.answering.formatting import format_answer
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.vision import generate_vision_answer,load_page_image,build_vision_content


@pytest.fixture
def client():
    return MagicMock()


def test_empty_question(client):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        answer_question("", [], client, "test-model")


def test_no_pages_uses_general_knowledge(client):
    with patch(
        "doubtdesk.answering.service.generate_general_answer",
        return_value="Water is H2O.",
    ) as mock_general:
        result = answer_question(
            "What is water?", [], client, "test-model"
        )

    assert result == {
        "status": "general_knowledge",
        "answer": "Water is H2O.",
        "sources": [],
    }

    mock_general.assert_called_once()


def test_grounded_answer(client):
    source = MagicMock()
    source.doc_id = "doc_001"
    source.page_numbers = [2]
    source.model_dump.return_value = {
        "doc_id": "doc_001",
        "page_numbers": [2],
    }

    evidence = MagicMock()
    evidence.status = "sufficient"
    evidence.answer = "Photosynthesis converts light energy."
    evidence.sources = [source]

    pages = [{"doc_id": "doc_001", "page_number": 2}]

    result = resolve_answer(
        evidence, pages, "What is photosynthesis?", client, "test-model"
    )

    assert result["status"] == "grounded"
    assert result["answer"] == evidence.answer
    assert result["sources"] == [source.model_dump.return_value]


def test_insufficient_evidence_falls_back(client):
    evidence = MagicMock()
    evidence.status = "insufficient"
    evidence.answer = ""
    evidence.sources = []

    with patch(
        "doubtdesk.answering.service.generate_general_answer",
        return_value="General knowledge answer",
    ) as mock_general:
        result = resolve_answer(
            evidence,
            [{"doc_id": "doc_001", "page_number": 1}],
            "Explain gravity",
            client,
            "test-model",
        )

    assert result["status"] == "general_knowledge"
    assert result["sources"] == []
    mock_general.assert_called_once()


def test_format_grounded_answer():
    response = {
        "status": "grounded",
        "answer": "Photosynthesis uses sunlight.",
        "sources": [
            {"doc_id": "doc_001", "page_numbers": [3, 1, 3]},
        ],
    }

    result = format_answer(response, {"doc_001": "biology.pdf"})

    assert "Photosynthesis uses sunlight." in result
    assert "- biology.pdf: PDF page(s) 1, 3" in result


def test_format_general_answer():
    response = {
        "status": "general_knowledge",
        "answer": "Water is H2O.",
        "sources": [],
    }

    result = format_answer(response, {})

    assert result == "Water is H2O.\n\nNot from document."


def test_format_missing_filename():
    response = {
        "status": "grounded",
        "answer": "Answer",
        "sources": [{"doc_id": "missing", "page_numbers": [1]}],
    }

    with pytest.raises(ValueError, match="Document filename not found"):
        format_answer(response, {})


def test_format_unsupported_status():
    with pytest.raises(ValueError, match="Unsupported answer status"):
        format_answer(
            {"status": "unknown", "answer": "Answer", "sources": []},
            {},
        )


def test_general_answer(client):
    client.models.generate_content.return_value.text = "Gravity attracts masses."

    result = generate_general_answer(
        client, "test-model", "What is gravity?"
    )

    assert result == "Gravity attracts masses."
    client.models.generate_content.assert_called_once()


def test_general_answer_empty_response(client):
    client.models.generate_content.return_value.text = None

    with pytest.raises(RuntimeError, match="could not generate an answer"):
        generate_general_answer(client, "test-model", "What is gravity?")


def test_vision_empty_content(client):
    with pytest.raises(ValueError, match="Content cannot be empty"):
        generate_vision_answer(client, "test-model", [])

def test_load_page_image(tmp_path):
    image_path = tmp_path / "page_001.png"
    image_path.write_bytes(b"fake_png_data")

    result = load_page_image(image_path)

    assert result.inline_data.mime_type == "image/png"
    assert result.inline_data.data == b"fake_png_data"


def test_load_missing_page_image(tmp_path):
    image_path = tmp_path / "missing.png"

    with pytest.raises(FileNotFoundError, match="Page image not found"):
        load_page_image(image_path)


def test_build_vision_content(tmp_path):
    image_path = tmp_path / "page_001.png"
    image_path.write_bytes(b"fake_png_data")

    pages = [
        {
            "doc_id": "doc_001",
            "page_number": 3,
            "image_path": image_path,
        }
    ]

    content = build_vision_content(
        "What is photosynthesis?",
        pages,
    )

    assert len(content) == 3
    assert "What is photosynthesis?" in content[0]
    assert "Document ID: doc_001" in content[1]
    assert "PDF page number: 3" in content[1]
    assert content[2].inline_data.mime_type == "image/png"


@pytest.mark.parametrize("question", ["", "   "])
def test_build_vision_content_empty_question(question):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_vision_content(question, [{"doc_id": "doc_001"}])


def test_build_vision_content_no_pages():
    with pytest.raises(ValueError, match="At least one page is required"):
        build_vision_content("Explain gravity", [])

def test_generate_vision_answer(client):
    from doubtdesk.answering.schemas import EvidenceResult

    evidence = EvidenceResult(
        status="insufficient",
        answer="",
        sources=[],
    )

    client.models.generate_content.return_value.parsed = evidence

    result = generate_vision_answer(
        client,
        "test-model",
        ["Question and page content"],
    )

    assert result.status == "insufficient"
    assert result.answer == ""
    assert result.sources == []

    client.models.generate_content.assert_called_once()


def test_generate_vision_answer_invalid_response(client):
    client.models.generate_content.return_value.parsed = None

    with pytest.raises(
        RuntimeError,
        match="did not return a valid structured response",
    ):
        generate_vision_answer(
            client,
            "test-model",
            ["Question and page content"],
        )
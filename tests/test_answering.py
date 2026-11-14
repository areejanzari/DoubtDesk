import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from doubtdesk.answering.formatting import format_answer
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.vision import generate_vision_answer, build_vision_content, build_grounded_content


@pytest.fixture
def client():
    return MagicMock()


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
    client.responses.create.return_value.output_text = "Gravity attracts masses."

    result = generate_general_answer(
        client, "test-model", "What is gravity?"
    )

    assert result == "Gravity attracts masses."
    client.responses.create.assert_called_once()
    assert client.responses.create.call_args.kwargs["reasoning"] == {"effort": "none"}
    assert client.responses.create.call_args.kwargs["max_output_tokens"] == 300


def test_general_answer_empty_response(client):
    client.responses.create.return_value.output_text = None

    with pytest.raises(RuntimeError, match="Model returned no answer"):
        generate_general_answer(client, "test-model", "What is gravity?")


def test_vision_empty_content(client):
    with pytest.raises(ValueError, match="Content cannot be empty"):
        generate_vision_answer(client, "test-model", [])

def test_build_vision_content_encodes_image_and_page_metadata(tmp_path):
    import base64

    image_path = tmp_path / "page_001.png"
    image_bytes = b"fake_png_data"
    image_path.write_bytes(image_bytes)
    pages = [{"doc_id": "doc_001", "page_number": 3, "image_path": image_path}]

    content = build_vision_content("What is photosynthesis?", pages)

    assert len(content) == 3
    assert content[0] == {"type": "input_text", "text": "Student question: What is photosynthesis?"}
    assert content[1]["type"] == "input_text"
    assert "Document ID: doc_001" in content[1]["text"]
    assert "PDF page number: 3" in content[1]["text"]
    assert content[2]["type"] == "input_image"
    assert content[2]["detail"] == "high"
    assert content[2]["image_url"] == "data:image/png;base64," + base64.b64encode(image_bytes).decode()


def test_build_vision_content_missing_image(tmp_path):
    pages = [{"doc_id": "doc_001", "page_number": 3, "image_path": tmp_path / "missing.png"}]
    with pytest.raises(FileNotFoundError, match="Page image not found"):
        build_vision_content("What is photosynthesis?", pages)


@pytest.mark.parametrize("question", ["", "   "])
def test_build_vision_content_empty_question(question):
    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_vision_content(question, [{"doc_id": "doc_001"}])


def test_build_vision_content_no_pages():
    with pytest.raises(ValueError, match="At least one page is required"):
        build_vision_content("Explain gravity", [])


def test_build_grounded_content_combines_text_and_image(tmp_path):
    image_path = tmp_path / "page_003.png"
    image_path.write_bytes(b"png")
    content = build_grounded_content(
        "Explain the nephron",
        [{"doc_id": "doc_001", "page_number": 3, "text": "Bowman's capsule"}],
        [{"doc_id": "doc_001", "page_number": 3, "image_path": image_path}],
    )
    assert len(content) == 4
    assert "Bowman's capsule" in content[1]["text"]
    assert content[3]["type"] == "input_image"


def test_build_grounded_content_requires_evidence():
    with pytest.raises(ValueError, match="At least one retrieved chunk or page image"):
        build_grounded_content("Explain gravity", [], [])

def test_generate_vision_answer(client):
    from doubtdesk.answering.schemas import EvidenceResult

    evidence = EvidenceResult(
        status="insufficient",
        answer="",
        sources=[],
    )

    client.responses.parse.return_value.output_parsed = evidence

    result = generate_vision_answer(
        client,
        "test-model",
        ["Question and page content"],
    )

    assert result.status == "insufficient"
    assert result.answer == ""
    assert result.sources == []

    client.responses.parse.assert_called_once()
    assert client.responses.parse.call_args.kwargs["reasoning"] == {"effort": "none"}
    assert client.responses.parse.call_args.kwargs["max_output_tokens"] == 700


def test_generate_vision_answer_invalid_response(client):
    client.responses.parse.return_value.output_parsed = None

    with pytest.raises(
        RuntimeError,
        match="No structured evidence returned",
    ):
        generate_vision_answer(
            client,
            "test-model",
            ["Question and page content"],
        )
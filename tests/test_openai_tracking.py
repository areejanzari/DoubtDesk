import sqlite3
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.schemas import EvidenceResult
from doubtdesk.answering.vision import generate_vision_answer
from doubtdesk.analytics.store import initialize_analytics_db
from doubtdesk.analytics.tracker import record_openai_usage
from doubtdesk.guardrails.paraphrasing import resolve_and_paraphrase
from doubtdesk.observability.tracing import observe_llm_response


def test_general_answer_tracks_success():
    client = MagicMock()
    response = SimpleNamespace(output_text="Gravity attracts masses.")
    client.responses.create.return_value = response

    with patch("doubtdesk.answering.fallback.observe_llm_response") as track:
        answer = generate_general_answer(
            client, "test-model", "What is gravity?",
            request_id="request_001", conversation_id="chat_001",
        )

    assert answer == response.output_text
    track.assert_called_once_with(
        "general_answer", response, "test-model", "request_001", "chat_001"
    )
    assert client.responses.create.call_args.kwargs["reasoning"] == {"effort": "none"}


def test_general_answer_api_failure_does_not_record_success():
    client = MagicMock()
    client.responses.create.side_effect = TimeoutError("OpenAI timed out")
    with patch("doubtdesk.answering.fallback.observe_llm_response") as track:
        with pytest.raises(TimeoutError, match="OpenAI timed out"):
            generate_general_answer(client, "test-model", "What is gravity?", request_id="request_002")
    track.assert_not_called()


def test_general_answer_empty_response_is_rejected():
    client = MagicMock()
    response = SimpleNamespace(output_text="")
    client.responses.create.return_value = response
    with patch("doubtdesk.answering.fallback.observe_llm_response") as track:
        with pytest.raises(RuntimeError, match="Model returned no answer"):
            generate_general_answer(client, "test-model", "What is gravity?", request_id="request_003")
    track.assert_called_once_with("general_answer", response, "test-model", "request_003", None)


def test_paraphrasing_tracks_success():
    client = MagicMock()
    result = {
        "resolved_question": "What is photosynthesis?",
        "questions": [
            "What is photosynthesis?",
            "Explain photosynthesis.",
            "How do plants convert light energy?",
        ],
    }
    response = SimpleNamespace(output_parsed=result)
    client.responses.parse.return_value = response

    with patch("doubtdesk.guardrails.paraphrasing.observe_llm_response") as track:
        answer = resolve_and_paraphrase(
            client, "test-model", "What is photosynthesis?", [],
            request_id="request_004", conversation_id="chat_001",
        )

    assert answer.resolved_question == result["resolved_question"]
    assert answer.questions == result["questions"]
    track.assert_called_once_with("paraphrasing", response, "test-model", "request_004", "chat_001")
    assert client.responses.parse.call_args.kwargs["reasoning"] == {"effort": "none"}


def test_paraphrasing_missing_output_is_rejected():
    client = MagicMock()
    response = SimpleNamespace(output_parsed=None)
    client.responses.parse.return_value = response

    with patch("doubtdesk.guardrails.paraphrasing.observe_llm_response") as track:
        with pytest.raises(ValueError, match="Model returned no valid paraphrases"):
            resolve_and_paraphrase(
                client, "test-model", "What is photosynthesis?", [], request_id="request_005"
            )
    track.assert_called_once_with("paraphrasing", response, "test-model", "request_005", None)


def test_paraphrasing_api_failure_does_not_record_success():
    client = MagicMock()
    client.responses.parse.side_effect = TimeoutError("OpenAI timed out")
    with patch("doubtdesk.guardrails.paraphrasing.observe_llm_response") as track:
        with pytest.raises(TimeoutError, match="OpenAI timed out"):
            resolve_and_paraphrase(
                client, "test-model", "What is photosynthesis?", [], request_id="request_006"
            )
    track.assert_not_called()


def test_vision_tracks_success():
    client = MagicMock()
    evidence = EvidenceResult(status="insufficient", answer="", sources=[])
    response = SimpleNamespace(output_parsed=evidence)
    client.responses.parse.return_value = response

    with patch("doubtdesk.answering.vision.observe_llm_response") as track:
        answer = generate_vision_answer(
            client, "test-model", [{"type": "input_text", "text": "test-content"}],
            request_id="request_007", conversation_id="chat_001",
        )

    assert answer == evidence
    track.assert_called_once_with(
        "grounded_generation", response, "test-model", "request_007", "chat_001"
    )
    assert client.responses.parse.call_args.kwargs["reasoning"] == {"effort": "none"}


def test_vision_api_failure_does_not_record_success():
    client = MagicMock()
    client.responses.parse.side_effect = TimeoutError("OpenAI timed out")
    with patch("doubtdesk.answering.vision.observe_llm_response") as track:
        with pytest.raises(TimeoutError, match="OpenAI timed out"):
            generate_vision_answer(client, "test-model", ["test-content"], request_id="request_008")
    track.assert_not_called()


def test_vision_missing_structured_output_is_rejected():
    client = MagicMock()
    response = SimpleNamespace(output_parsed=None)
    client.responses.parse.return_value = response
    with patch("doubtdesk.answering.vision.observe_llm_response") as track:
        with pytest.raises(RuntimeError, match="No structured evidence returned"):
            generate_vision_answer(client, "test-model", ["test-content"], request_id="request_009")
    track.assert_called_once_with("grounded_generation", response, "test-model", "request_009", None)


def test_openai_response_usage_is_recorded(tmp_path):
    db_path = tmp_path / "analytics.db"
    initialize_analytics_db(db_path)
    response = SimpleNamespace(usage=SimpleNamespace(
        input_tokens=120,
        output_tokens=35,
        total_tokens=155,
        input_tokens_details=SimpleNamespace(cached_tokens=15),
        output_tokens_details=SimpleNamespace(reasoning_tokens=4),
    ))

    record_openai_usage(
        response=response,
        request_id="request_010",
        conversation_id="chat_001",
        operation="paraphrasing",
        model="test-model",
        latency_ms=90,
        db_path=db_path,
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT prompt_tokens, output_tokens, thinking_tokens, cached_tokens, total_tokens, latency_ms, status FROM ai_usage WHERE request_id = ?",
            ("request_010",),
        ).fetchone()
    assert row == (120, 35, 4, 15, 155, 90, "success")


def test_openai_usage_is_skipped_when_no_request_id():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5))
    with patch("doubtdesk.analytics.tracker.record_openai_usage") as track:
        observe_llm_response("general_answer", response, "test-model")
    track.assert_not_called()


def test_openai_usage_observer_forwards_request_metadata():
    response = SimpleNamespace(usage=SimpleNamespace(input_tokens=3, output_tokens=2, total_tokens=5))
    with patch("doubtdesk.analytics.tracker.record_openai_usage") as track:
        observe_llm_response("general_answer", response, "test-model", "request_012", "chat_001")
    track.assert_called_once_with(
        response=response,
        request_id="request_012",
        conversation_id="chat_001",
        operation="general_answer",
        model="test-model",
        latency_ms=None,
    )

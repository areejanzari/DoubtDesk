from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import pytest
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.guardrails.paraphrasing import resolve_and_paraphrase
from doubtdesk.answering.vision import generate_vision_answer


def test_general_answer_tracks_success():
    client = MagicMock()
    response = SimpleNamespace(
        text="Gravity is a force of attraction.",
        usage_metadata=SimpleNamespace(
            total_token_count=150,
        ),
    )

    client.models.generate_content.return_value = response

    with patch(
        "doubtdesk.answering.fallback.record_gemini_usage"
    ) as mock_tracker:
        answer = generate_general_answer(
            client=client,
            model_name="test-model",
            question="What is gravity?",
            request_id="request_001",
            conversation_id="chat_001",
        )

    assert answer == "Gravity is a force of attraction."

    mock_tracker.assert_called_once()

    kwargs = mock_tracker.call_args.kwargs

    assert kwargs["request_id"] == "request_001"
    assert kwargs["operation"] == "general_answer"
    assert kwargs["response"] is response
    assert kwargs["latency_ms"] >= 0


def test_general_answer_tracks_api_failure():
    client = MagicMock()
    client.models.generate_content.side_effect = TimeoutError(
        "Gemini request timed out"
    )

    with patch(
        "doubtdesk.answering.fallback.record_gemini_failure"
    ) as mock_tracker:
        with pytest.raises(TimeoutError):
            generate_general_answer(
                client=client,
                model_name="test-model",
                question="What is gravity?",
                request_id="request_002",
                conversation_id="chat_001",
            )

    mock_tracker.assert_called_once()

    kwargs = mock_tracker.call_args.kwargs

    assert kwargs["operation"] == "general_answer"
    assert kwargs["error_type"] == "TimeoutError"
    assert kwargs["request_id"] == "request_002"

def test_general_answer_tracks_validation_failure():
    client = MagicMock()

    response = SimpleNamespace(
        text="",
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            total_token_count=150,
        ),
    )

    client.models.generate_content.return_value = response

    with (
        patch(
            "doubtdesk.answering.fallback.record_gemini_failure"
        ) as mock_failure,
        patch(
            "doubtdesk.answering.fallback.record_gemini_usage"
        ) as mock_success,
    ):
        with pytest.raises(
            RuntimeError,
            match="The AI model could not generate an answer",
        ):
            generate_general_answer(
                client=client,
                model_name="test-model",
                question="What is gravity?",
                request_id="request_003",
                conversation_id="chat_001",
            )

    mock_failure.assert_called_once()
    mock_success.assert_not_called()

    kwargs = mock_failure.call_args.kwargs

    assert kwargs["request_id"] == "request_003"
    assert kwargs["operation"] == "general_answer"
    assert kwargs["error_type"] == "RuntimeError"
    assert kwargs["response"] is response

def test_paraphrasing_tracks_success():
    client = MagicMock()

    response = SimpleNamespace(
        parsed={
            "resolved_question": "What is photosynthesis?",
            "questions": [
                "What is photosynthesis?",
                "Explain the process of photosynthesis.",
                "How do plants convert light into chemical energy?",
            ],
        },
        usage_metadata=SimpleNamespace(
            total_token_count=250,
        ),
    )

    client.models.generate_content.return_value = response

    with patch(
        "doubtdesk.guardrails.paraphrasing.record_gemini_usage"
    ) as mock_success:
        result = resolve_and_paraphrase(
            client=client,
            model_name="test-model",
            question="What is photosynthesis?",
            history=[],
            request_id="request_004",
            conversation_id="chat_001",
        )

    assert len(result.questions) == 3
    assert result.resolved_question == "What is photosynthesis?"

    mock_success.assert_called_once()

    kwargs = mock_success.call_args.kwargs

    assert kwargs["operation"] == "paraphrasing"
    assert kwargs["request_id"] == "request_004"
    assert kwargs["response"] is response

def test_paraphrasing_tracks_validation_failure():
    client = MagicMock()

    response = SimpleNamespace(
        parsed=None,
        usage_metadata=SimpleNamespace(
            total_token_count=180,
        ),
    )

    client.models.generate_content.return_value = response

    with (
        patch(
            "doubtdesk.guardrails.paraphrasing.record_gemini_failure"
        ) as mock_failure,
        patch(
            "doubtdesk.guardrails.paraphrasing.record_gemini_usage"
        ) as mock_success,
    ):
        with pytest.raises(
            ValueError,
            match="Gemini did not return valid paraphrases",
        ):
            resolve_and_paraphrase(
                client=client,
                model_name="test-model",
                question="What is photosynthesis?",
                history=[],
                request_id="request_005",
                conversation_id="chat_001",
            )

    mock_failure.assert_called_once()
    mock_success.assert_not_called()

    kwargs = mock_failure.call_args.kwargs

    assert kwargs["operation"] == "paraphrasing"
    assert kwargs["error_type"] == "ValueError"
    assert kwargs["response"] is response

def test_paraphrasing_tracks_api_failure():
    client = MagicMock()
    client.models.generate_content.side_effect = TimeoutError(
        "Gemini request timed out"
    )

    with (
        patch(
            "doubtdesk.guardrails.paraphrasing.record_gemini_failure"
        ) as mock_failure,
        patch(
            "doubtdesk.guardrails.paraphrasing.record_gemini_usage"
        ) as mock_success,
    ):
        with pytest.raises(TimeoutError):
            resolve_and_paraphrase(
                client=client,
                model_name="test-model",
                question="What is photosynthesis?",
                history=[],
                request_id="request_006",
                conversation_id="chat_001",
            )

    mock_failure.assert_called_once()
    mock_success.assert_not_called()

    kwargs = mock_failure.call_args.kwargs

    assert kwargs["operation"] == "paraphrasing"
    assert kwargs["error_type"] == "TimeoutError"
    assert kwargs["request_id"] == "request_006"

def test_vision_tracks_success():
    client = MagicMock()

    response = SimpleNamespace(
        parsed={
            "status": "insufficient",
            "answer": "",
            "sources": [],
        },
        usage_metadata=SimpleNamespace(
            total_token_count=1500,
        ),
    )

    client.models.generate_content.return_value = response

    with patch(
        "doubtdesk.answering.vision.record_gemini_usage"
    ) as mock_success:
        result = generate_vision_answer(
            client=client,
            model_name="test-model",
            content=["test-content"],
            request_id="request_007",
            conversation_id="chat_001",
        )

    assert result.status == "insufficient"
    assert result.answer == ""

    mock_success.assert_called_once()

    kwargs = mock_success.call_args.kwargs

    assert kwargs["operation"] == "vision"
    assert kwargs["request_id"] == "request_007"
    assert kwargs["response"] is response


def test_vision_tracks_api_failure():
    client = MagicMock()
    client.models.generate_content.side_effect = TimeoutError(
        "Gemini vision request timed out"
    )

    with (
        patch(
            "doubtdesk.answering.vision.record_gemini_failure"
        ) as mock_failure,
        patch(
            "doubtdesk.answering.vision.record_gemini_usage"
        ) as mock_success,
    ):
        with pytest.raises(TimeoutError):
            generate_vision_answer(
                client=client,
                model_name="test-model",
                content=["test-content"],
                request_id="request_008",
                conversation_id="chat_001",
            )

    mock_failure.assert_called_once()
    mock_success.assert_not_called()

    kwargs = mock_failure.call_args.kwargs

    assert kwargs["operation"] == "vision"
    assert kwargs["error_type"] == "TimeoutError"
    assert kwargs["request_id"] == "request_008"

def test_vision_tracks_validation_failure():
    client = MagicMock()

    response = SimpleNamespace(
        parsed=None,
        usage_metadata=SimpleNamespace(
            total_token_count=1200,
        ),
    )

    client.models.generate_content.return_value = response

    with (
        patch(
            "doubtdesk.answering.vision.record_gemini_failure"
        ) as mock_failure,
        patch(
            "doubtdesk.answering.vision.record_gemini_usage"
        ) as mock_success,
    ):
        with pytest.raises(
            RuntimeError,
            match="The AI model did not return a valid structured response",
        ):
            generate_vision_answer(
                client=client,
                model_name="test-model",
                content=["test-content"],
                request_id="request_009",
                conversation_id="chat_001",
            )

    mock_failure.assert_called_once()
    mock_success.assert_not_called()

    kwargs = mock_failure.call_args.kwargs

    assert kwargs["operation"] == "vision"
    assert kwargs["error_type"] == "RuntimeError"
    assert kwargs["response"] is response
    
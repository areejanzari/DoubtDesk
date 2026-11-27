import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from doubtdesk.answering.evidence_selection import (
    TextEvidenceAssessment, choose_evidence_route,
)
from doubtdesk.answering.fallback import ANSWER_MAX_OUTPUT_TOKENS
from doubtdesk.answering.output_guardrails import (
    DOCUMENT_FALLBACK_MESSAGE,
    BedrockOpenAIOutputJudge, JudgeVerdict, document_insufficient_result,
    is_document_specific, review_and_repair,
)
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.vision import generate_vision_answer


CHUNKS = [{"doc_id": "docA", "page_number": 1, "text": "Photosynthesis stores light energy", "chunk_id": "c1"}]


def route(*values):
    paraphrases = [f"paraphrase-{index}" for index in range(len(values))]
    assessment = TextEvidenceAssessment(sufficiency_score=sum(values)/len(values))
    return choose_evidence_route(assessment, paraphrases, CHUNKS)


@pytest.mark.parametrize("scores,tier,mode,images", [
    ([0.80, 0.90], "high", "text", 0),
    ([0.50], "moderate", "visual", 1),
    ([0.79, 0.98], "high", "text", 0),
    ([0.1, 0.9], "moderate", "visual", 1),
    ([0.99, 0.01, 0.01], "low", "low", 0),
    ([0.90, 0.90, 0.10], "moderate", "visual", 1),
    ([0.85, 1.00, 0.50], "moderate", "visual", 1),
    ([0.49], "low", "low", 0),
    ([0.01, 0.20], "low", "low", 0),
])
def test_three_tiers(scores, tier, mode, images):
    selected = route(*scores)
    assert selected["evidence_tier"] == tier
    assert selected["evidence_mode"] == mode
    assert len(selected["retrieved_pages"]) == images
    assert selected["evidence_average"] == pytest.approx(sum(scores) / len(scores))


def test_no_evidence_cannot_pass_even_with_high_score():
    assessment = TextEvidenceAssessment(sufficiency_score=0.98)
    assert choose_evidence_route(assessment, ["Expected"], [])["evidence_mode"] == "low"


def test_document_specific_questions_are_detected():
    assert is_document_specific("In the uploaded document, what is on page 1?")
    assert not is_document_specific("According to Newton's law, why does this happen?")


def verdict(passed=True, feedback="Revise for completeness"):
    return JudgeVerdict(
        passed=passed, relevance=0.95 if passed else 0.4,
        groundedness=0.9, completeness=0.9,
        safety="pass", feedback=feedback,
    )


class FakeJudge:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        response = self.answers.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_judge_review_approved_without_regeneration():
    judge = FakeJudge(verdict())
    result = review_and_repair(
        result={"status": "general_knowledge", "answer": "Accurate answer", "sources": []},
        question="Question", judge=judge,
        regenerate=lambda _: pytest.fail("Unexpected regeneration"),
    )
    assert result["judge_passed"] is True
    assert result["judge_attempts"] == 1
    assert len(judge.calls) == 1


def test_judge_rejected_then_openai_regenerated_once():
    judge = FakeJudge(verdict(False, "Correct this fact"), verdict())
    feedback = []

    def regenerate(message):
        feedback.append(message)
        return {"status": "general_knowledge", "answer": "Fixed answer", "sources": []}

    result = review_and_repair(
        result={"status": "general_knowledge", "answer": "Wrong fact", "sources": []},
        question="Question", judge=judge, regenerate=regenerate,
    )
    assert result["answer"] == "Fixed answer"
    assert result["judge_passed"] is True
    assert result["judge_attempts"] == 2
    assert feedback == ["Correct this fact"]
    assert len(judge.calls) == 2


def test_failed_judge_and_retry_never_release_candidate():
    judge = FakeJudge(verdict(False), verdict(False))
    result = review_and_repair(
        result={"status": "general_knowledge", "answer": "Bad answer", "sources": []},
        question="Question", judge=judge,
        regenerate=lambda feedback: {"status": "general_knowledge", "answer": "Still bad", "sources": []},
    )
    assert result["judge_passed"] is False
    assert "Bad answer" not in result["answer"]
    assert "Still bad" not in result["answer"]
    assert len(judge.calls) == 2


def test_judge_exception_fails_closed_without_regeneration():
    judge = FakeJudge(RuntimeError("Judge unavailable"))
    result = review_and_repair(
        result={"status": "general_knowledge", "answer": "Possible hallucination", "sources": []},
        question="Question", judge=judge,
        regenerate=lambda _: pytest.fail("Never regenerate on judge outage"),
    )
    assert result["judge_passed"] is False
    assert result["answer"] != "Possible hallucination"


def test_invalid_citations_cannot_be_approved_by_judge():
    judge = FakeJudge(verdict())
    result = review_and_repair(
        result={"status": "grounded", "answer": "A", "sources": [{"doc_id": "wrong", "page_numbers": [1]}]},
        question="Question", judge=judge,
        regenerate=lambda _: {"status": "grounded", "answer": "B", "sources": [{"doc_id": "wrong", "page_numbers": [2]}]},
        allowed_pages=[{"doc_id": "docA", "page_number": 1}],
    )
    assert result["judge_passed"] is False
    assert not judge.calls


def chat_reply(payload):
    message = SimpleNamespace(content=json.dumps(payload))
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)], usage=None)


def test_bedrock_judge_client_is_called():
    client = MagicMock()
    client.chat.completions.create.return_value = chat_reply(verdict().model_dump())
    pacer = MagicMock()
    judge = BedrockOpenAIOutputJudge(client, model_name="qwen-judge", pacer=pacer)
    verdict_result = judge.evaluate(
        question="What is ATP?",
        result={"status": "grounded", "answer": "ATP stores energy", "sources": [{"doc_id": "docA", "page_numbers": [1]}]},
        chunks=CHUNKS,
    )
    assert verdict_result.passed
    assert client.chat.completions.create.call_count == 1
    assert client.chat.completions.create.call_args.kwargs["model"] == "qwen-judge"
    pacer.acquire.assert_called_once()


def test_judge_cannot_claim_pass_when_scores_are_low():
    client = MagicMock()
    client.chat.completions.create.return_value = chat_reply({
        "passed": True, "relevance": 0.1, "groundedness": 0.9,
        "completeness": 0.9, "safety": "pass", "feedback": "Off topic",
    })
    judge = BedrockOpenAIOutputJudge(client, model_name="m", pacer=MagicMock())
    response = judge.evaluate(question="Question", result={"status": "general_knowledge", "answer": "Answer", "sources": []})
    assert not response.passed


def test_generation_function_output_caps():
    client = MagicMock()
    client.responses.create.return_value = SimpleNamespace(output_text="Answer")
    generate_general_answer(client, "openai-model", "Question")
    assert client.responses.create.call_args.kwargs["max_output_tokens"] == ANSWER_MAX_OUTPUT_TOKENS

    client.responses.parse.return_value = SimpleNamespace(output_parsed={
        "status": "insufficient", "answer": "", "sources": [],
    })
    generate_vision_answer(client, "openai-model", [{"type": "input_text", "text": "Question"}])
    assert client.responses.parse.call_args.kwargs["max_output_tokens"] > ANSWER_MAX_OUTPUT_TOKENS




def test_no_llm_for_pdf_specific_low_evidence():
    result = document_insufficient_result()
    assert result["answer"] == DOCUMENT_FALLBACK_MESSAGE
    assert result["skip_output_judge"]








def test_judge_http_503_is_not_a_grounding_failure():
    error = RuntimeError("This model is currently experiencing high demand")
    error.code = 503
    judge = FakeJudge(error)
    result = review_and_repair(
        result={"status": "grounded", "answer": "Nephron has a glomerulus.",
                "sources": [{"doc_id": "docA", "page_numbers": [1]}]},
        question="Describe the structure of a nephron",
        judge=judge,
        regenerate=lambda _: pytest.fail("Do not regenerate when judge is unavailable"),
        allowed_pages=[{"doc_id": "docA", "page_number": 1}],
    )
    assert result["judge_attempts"] == 1
    assert result["judge_regenerations"] == 0
    assert result["judge_reviews"][0]["http_status"] == 503
    assert result["judge_failure_reason"] == "judge_unavailable"
    assert "verification is temporarily unavailable" in result["answer"]
    assert "couldn't find enough evidence" not in result["answer"]
    assert result["status"] == "verification_failed"
    assert result["judge_needs_fallback"] is False


def test_optional_judge_model_fallback_on_503(monkeypatch):
    monkeypatch.setenv("OUTPUT_JUDGE_FALLBACK_MODEL", "backup-judge-model")
    error = RuntimeError("High demand")
    error.code = 503
    client = MagicMock()
    client.chat.completions.create.side_effect = [error, chat_reply(verdict().model_dump())]
    pacer = MagicMock()
    judge = BedrockOpenAIOutputJudge(client, model_name="primary-judge-model", pacer=pacer)
    result = judge.evaluate(
        question="What is ATP?",
        result={"status": "general_knowledge", "answer": "ATP stores energy", "sources": []},
    )
    assert result.passed
    assert [call.kwargs["model"] for call in client.chat.completions.create.call_args_list] == [
        "primary-judge-model", "backup-judge-model",
    ]
    assert pacer.acquire.call_count == 2


def test_verification_failure_is_not_formatted_as_general_knowledge(tmp_path):
    from doubtdesk.api.streaming import finalize_stream_result

    message = "Answer verification is temporarily unavailable. Please try again."
    state = {"status": "verification_failed", "answer": message, "sources": [], "verification_failed": True}
    streamed = finalize_stream_result(dict(state))
    assert streamed["formatted_answer"] == message


def test_single_score_drives_routing():
    assert route(0.33)["evidence_tier"] == "low"
    assert route(0.81)["evidence_tier"] == "high"
    assert route(0.78)["evidence_average"] == pytest.approx(0.78)


def test_failed_api_call_is_distinct_from_rejected_answer():
    exc = RuntimeError("This model is experiencing high demand")
    exc.code = 503
    result = review_and_repair(
        result={"status": "grounded", "answer": "Renal tubule has several segments.",
                "sources": [{"doc_id": "docA", "page_numbers": [1]}]},
        question="Describe a nephron", judge=FakeJudge(exc),
        regenerate=lambda _: pytest.fail("Should not regenerate on 503"),
        allowed_pages=[{"doc_id": "docA", "page_number": 1}],
    )
    assert result["status"] == "verification_failed"
    assert result["judge_failure_reason"] == "judge_unavailable"
    assert result["judge_attempts"] == 1
    assert result["judge_regenerations"] == 0
    assert result["judge_needs_fallback"] is False
    assert result["judge_reviews"][0]["http_status"] == 503




def test_verification_failure_api_schema_and_formatting():
    from doubtdesk.api.schemas import AskResponse
    from doubtdesk.answering.formatting import format_answer
    result = {"status": "verification_failed", "answer": "Verification unavailable", "sources": [],
              "formatted_answer": "Verification unavailable"}
    response = AskResponse.model_validate(result)
    assert response.status == "verification_failed"
    assert format_answer(result, {}) == "Verification unavailable"


def test_hybrid_retrieval_top_k_config(monkeypatch):
    from doubtdesk.retrieval.hybrid import hybrid_search
    monkeypatch.setenv("RETRIEVAL_TOP_K", "6")
    dense = [[
        {"chunk_id": "c" + str(i), "doc_id": "docA", "page_number": i + 1,
         "text": "text" + str(i), "distance": 0.1 + 0.01*i}
        for i in range(10)
    ]]
    with (
        patch("doubtdesk.retrieval.hybrid.embed_queries", return_value=[0.1]),
        patch("doubtdesk.retrieval.hybrid.search_chunks", return_value=dense),
        patch("doubtdesk.retrieval.hybrid.lexical_candidates", return_value=[]),
    ):
        result = hybrid_search(None, ["What is photosynthesis?"], ["docA"])
    assert len(result) == 6
    assert len({item["chunk_id"] for item in result}) == 6

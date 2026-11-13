from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from doubtdesk.answering.evidence_selection import (
    ParaphraseEvidence, TextEvidenceAssessment, choose_evidence_route,
)
from doubtdesk.answering.output_guardrails import (
    ANSWER_MAX_OUTPUT_TOKENS, DOCUMENT_FALLBACK_MESSAGE,
    GeminiOutputJudge, JudgeVerdict, document_insufficient_result,
    is_document_specific, review_and_repair,
)
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.vision import generate_vision_answer
from doubtdesk.graph.nodes import route_selected_evidence, output_guardrail_node
from doubtdesk.graph.builder import build_graph


CHUNKS = [{"doc_id": "docA", "page_number": 1, "text": "Photosynthesis stores light energy", "chunk_id": "c1"}]


def route(*values):
    paraphrases = [f"paraphrase-{index}" for index in range(len(values))]
    assessment = TextEvidenceAssessment(scores=[
        ParaphraseEvidence(paraphrase=question, score=score, explanation="x")
        for question, score in zip(paraphrases, values, strict=True)
    ])
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


def test_mismatched_paraphrase_scores_fail_closed():
    assessment = TextEvidenceAssessment(scores=[ParaphraseEvidence(
        paraphrase="Wrong paraphrase", score=0.98, explanation="Mismatch",
    )])
    assert choose_evidence_route(assessment, ["Expected"], CHUNKS)["evidence_mode"] == "low"


def test_low_tier_is_routed_without_image_or_grounded_generation():
    assert route_selected_evidence({"evidence_mode": "low", "question": "Explain thermodynamics"}) == "fallback"
    assert route_selected_evidence({"evidence_mode": "low", "question": "According to this PDF, what is thermodynamics?"}) == "insufficient_document"
    assert route_selected_evidence({"evidence_mode": "text", "question": "Explain thermodynamics"}) == "evidence"
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


def test_gemini_review_approved_without_regeneration():
    judge = FakeJudge(verdict())
    result = review_and_repair(
        result={"status": "general_knowledge", "answer": "Accurate answer", "sources": []},
        question="Question", judge=judge,
        regenerate=lambda _: pytest.fail("Unexpected regeneration"),
    )
    assert result["judge_passed"] is True
    assert result["judge_attempts"] == 1
    assert len(judge.calls) == 1


def test_gemini_rejected_then_openai_regenerated_once():
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
    judge = FakeJudge(RuntimeError("Gemini unavailable"))
    result = review_and_repair(
        result={"status": "general_knowledge", "answer": "Possible hallucination", "sources": []},
        question="Question", judge=judge,
        regenerate=lambda _: pytest.fail("Never regenerate on judge outage"),
    )
    assert result["judge_passed"] is False
    assert result["answer"] != "Possible hallucination"


def test_invalid_citations_cannot_be_approved_by_gemini():
    judge = FakeJudge(verdict())
    result = review_and_repair(
        result={"status": "grounded", "answer": "A", "sources": [{"doc_id": "wrong", "page_numbers": [1]}]},
        question="Question", judge=judge,
        regenerate=lambda _: {"status": "grounded", "answer": "B", "sources": [{"doc_id": "wrong", "page_numbers": [2]}]},
        allowed_pages=[{"doc_id": "docA", "page_number": 1}],
    )
    assert result["judge_passed"] is False
    assert not judge.calls


def test_gemini_sdk_client_is_called():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(parsed=verdict().model_dump())
    pacer = MagicMock()
    judge = GeminiOutputJudge(client, model_name="gemini-3.1-flash-lite", pacer=pacer)
    verdict_result = judge.evaluate(
        question="What is ATP?",
        result={"status": "grounded", "answer": "ATP stores energy", "sources": [{"doc_id": "docA", "page_numbers": [1]}]},
        chunks=CHUNKS,
    )
    assert verdict_result.passed
    assert client.models.generate_content.call_count == 1
    assert client.models.generate_content.call_args.kwargs["model"] == "gemini-3.1-flash-lite"
    pacer.acquire.assert_called_once()


def test_gemini_cannot_claim_pass_when_scores_are_low():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(parsed={
        "passed": True, "relevance": 0.1, "groundedness": 0.9,
        "completeness": 0.9, "safety": "pass", "feedback": "Off topic",
    })
    judge = GeminiOutputJudge(client, pacer=MagicMock())
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
    assert client.responses.parse.call_args.kwargs["max_output_tokens"] == ANSWER_MAX_OUTPUT_TOKENS


def test_graph_has_live_output_judge_node(tmp_path):
    graph = build_graph(
        collection=MagicMock(), embedding_model=MagicMock(),
        client=MagicMock(), model_name="test-model",
        scope_classifier=MagicMock(), db_path=tmp_path / "documents.db",
        output_judge=FakeJudge(verdict()),
    )
    assert "output_guardrail" in graph.get_graph().nodes
    assert "insufficient_document" in graph.get_graph().nodes


def test_no_llm_for_pdf_specific_low_evidence():
    result = document_insufficient_result()
    assert result["answer"] == DOCUMENT_FALLBACK_MESSAGE
    assert result["skip_output_judge"]


def test_sync_graph_judge_calls_and_corrects(monkeypatch):
    judge = FakeJudge(verdict(False, "Fix the claim"), verdict())
    answers = iter(["Fixed answer"])
    monkeypatch.setattr("doubtdesk.graph.nodes.generate_general_answer", lambda *args, **kwargs: next(answers))
    review = output_guardrail_node(MagicMock(), "test", judge)
    result = review({"question": "Why?", "status": "general_knowledge", "answer": "Bad answer", "sources": []})
    assert result["answer"] == "Fixed answer"
    assert result["judge_passed"] is True
    assert len(judge.calls) == 2


def test_main_graph_general_answer_actually_calls_output_judge(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "doubtdesk.graph.nodes.resolve_and_paraphrase",
        lambda **kw: SimpleNamespace(resolved_question=kw["question"], questions=[kw["question"]] * 3),
    )
    with patch("doubtdesk.graph.nodes.generate_general_answer", return_value="Gravity attracts masses"):
        classifier = MagicMock()
        classifier.predict.return_value = [{"label": 0}] * 3
        judge = FakeJudge(verdict())
        graph = build_graph(
            collection=MagicMock(), embedding_model=MagicMock(), client=MagicMock(),
            model_name="gpt-test", scope_classifier=classifier,
            db_path=tmp_path / "empty.db", output_judge=judge,
        )
        result = graph.invoke({"question": "What is gravity?", "doc_ids": [], "user_id": "u"})
    assert result["judge_passed"] is True
    assert result["answer"] == "Gravity attracts masses"
    assert len(judge.calls) == 1


def test_main_graph_grounded_answer_actually_calls_output_judge(tmp_path, monkeypatch):
    from doubtdesk.answering.schemas import EvidenceResult

    monkeypatch.setattr(
        "doubtdesk.graph.nodes.resolve_and_paraphrase",
        lambda **kw: SimpleNamespace(resolved_question=kw["question"], questions=[kw["question"]] * 3),
    )
    evidence = EvidenceResult(
        status="sufficient", answer="Plants use light energy",
        sources=[{"doc_id": "docA", "page_numbers": [1]}],
    )
    with (
        patch("doubtdesk.graph.nodes.validate_user_document_ids"),
        patch("doubtdesk.graph.nodes.hybrid_search", return_value=CHUNKS),
        patch("doubtdesk.graph.nodes.assess_text_evidence", return_value=TextEvidenceAssessment(
            scores=[ParaphraseEvidence(paraphrase="Explain photosynthesis", score=0.95, explanation="Supported") for _ in range(3)],
        )),
        patch("doubtdesk.graph.nodes.generate_vision_answer", return_value=evidence),
        patch("doubtdesk.graph.nodes.get_document_names", return_value={"docA": "bio.pdf"}),
    ):
        classifier = MagicMock()
        classifier.predict.return_value = [{"label": 0}] * 3
        judge = FakeJudge(verdict())
        graph = build_graph(
            collection=MagicMock(), embedding_model=MagicMock(), client=MagicMock(),
            model_name="gpt-test", scope_classifier=classifier,
            db_path=tmp_path / "empty.db", output_judge=judge,
        )
        result = graph.invoke({
            "question": "Explain photosynthesis", "doc_ids": ["docA"], "user_id": "u",
        })
    assert result["status"] == "grounded"
    assert result["judge_passed"] is True
    assert len(judge.calls) == 1
    assert judge.calls[0]["result"]["sources"][0]["doc_id"] == "docA"


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


def test_optional_gemini_model_fallback_on_503(monkeypatch):
    monkeypatch.setenv("OUTPUT_JUDGE_FALLBACK_MODEL", "backup-gemini-model")
    error = RuntimeError("High demand")
    error.code = 503
    client = MagicMock()
    client.models.generate_content.side_effect = [error, SimpleNamespace(parsed=verdict().model_dump())]
    pacer = MagicMock()
    judge = GeminiOutputJudge(client, model_name="primary-gemini-model", pacer=pacer)
    result = judge.evaluate(
        question="What is ATP?",
        result={"status": "general_knowledge", "answer": "ATP stores energy", "sources": []},
    )
    assert result.passed
    assert [call.kwargs["model"] for call in client.models.generate_content.call_args_list] == [
        "primary-gemini-model", "backup-gemini-model",
    ]
    assert pacer.acquire.call_count == 2


def test_verification_failure_is_not_formatted_as_general_knowledge(tmp_path):
    import sqlite3
    from doubtdesk.graph.nodes import formatting_node
    from doubtdesk.api.streaming import finalize_stream_result

    message = "Answer verification is temporarily unavailable. Please try again."
    state = {"status": "verification_failed", "answer": message, "sources": [], "verification_failed": True}
    formatted = formatting_node(tmp_path / "documents.db")(state)
    assert formatted["formatted_answer"] == message
    streamed = finalize_stream_result(dict(state), db_path=tmp_path / "documents.db")
    assert streamed["formatted_answer"] == message


def test_average_routing_uses_mean_even_when_one_requirement_is_very_high():
    assert route(0.99, 0.01, 0.01)["evidence_tier"] == "low"
    assert route(0.94, 0.92, 0.54)["evidence_tier"] == "high"
    assert route(0.85, 1.0, 0.5)["evidence_average"] == pytest.approx(0.7833333333333333)


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


def test_output_judge_rejection_on_grounding_routes_to_real_fallback():
    from doubtdesk.graph.nodes import route_after_output_guardrail, fallback_node

    judge = FakeJudge(verdict(False), verdict(False))
    reviewer = output_guardrail_node(MagicMock(), "test", judge)
    with patch("doubtdesk.graph.nodes.generate_vision_answer") as regenerate:
        regenerate.return_value = SimpleNamespace(
            status="sufficient", answer="Still unverified", sources=[SimpleNamespace(
                doc_id="docA", page_numbers=[1], model_dump=lambda: {"doc_id": "docA", "page_numbers": [1]}
            )]
        )
        result = reviewer({
            "question": "What does a nephron do?", "status": "grounded", "answer": "Unverified answer",
            "sources": [{"doc_id": "docA", "page_numbers": [1]}],
            "retrieved_chunks": CHUNKS, "retrieved_pages": [],
            "evidence_pages": [{"doc_id": "docA", "page_number": 1}],
        })
    assert route_after_output_guardrail(result) == "fallback"
    assert result["judge_needs_fallback"] is True
    with patch("doubtdesk.graph.nodes.generate_general_answer", return_value="General answer"):
        fallback = fallback_node(MagicMock(), "test")
        followup = fallback({"question": "What does a nephron do?", **result})
    assert followup["status"] == "general_knowledge"
    assert followup["judge_fallback_attempted"] is True
    assert followup["answer"] == "General answer"


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
    dense = {
        "ids": [["c" + str(i) for i in range(10)]],
        "documents": [["text" + str(i) for i in range(10)]],
        "metadatas": [[{"doc_id": "docA", "page_number": i + 1} for i in range(10)]],
        "distances": [[0.1 + 0.01*i for i in range(10)]],
    }
    with (
        patch("doubtdesk.retrieval.hybrid.embed_queries", return_value=[0.1]),
        patch("doubtdesk.retrieval.hybrid.search_chunks", return_value=dense),
        patch("doubtdesk.retrieval.hybrid.lexical_candidates", return_value=[]),
    ):
        result = hybrid_search(None, None, ["What is photosynthesis?"], ["docA"])
    assert len(result) == 6
    assert len({item["chunk_id"] for item in result}) == 6

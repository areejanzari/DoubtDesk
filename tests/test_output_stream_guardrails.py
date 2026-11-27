from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from doubtdesk.api.streaming import stream_answer
from doubtdesk.answering.output_guardrails import JudgeVerdict


class FakeStream:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.output_text.delta", delta=self.text)

    def get_final_response(self):
        return SimpleNamespace(usage=None)


class FakeOpenAI:
    def __init__(self, text):
        self.text = text
        self.calls = []
        self.responses = self

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.text)


class FakeJudge:
    def __init__(self, verdicts):
        self.verdicts = iter(verdicts)
        self.calls = 0

    def evaluate(self, **kwargs):
        self.calls += 1
        return next(self.verdicts)


def verdict(passed):
    return JudgeVerdict(
        passed=passed,
        relevance=1.0 if passed else 0.3,
        completeness=1.0,
        groundedness=1.0,
        safety="pass",
        feedback="Repair the answer" if not passed else "",
    )


def visible_text(events):
    last_reset = max((index for index, event in enumerate(events) if event[0] == "reset"), default=-1)
    return "".join(event[1]["text"] for event in events[last_reset + 1:] if event[0] == "token")


def collect(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as finish:
            return events, finish.value


def test_stream_tokens_arrive_before_judge_check_and_are_kept_on_approval():
    openai = FakeOpenAI("Correct response")
    judge = FakeJudge([verdict(True)])
    events, result = collect(stream_answer({"question": "What is gravity?", "doc_ids": []}, openai, "openai-model", judge=judge))
    kinds = [event[0] for event in events]
    assert events[0] == ("status", {"stage": "verifying"})
    assert kinds.index("token") < events.index(("status", {"stage": "checking"}))
    assert "reset" not in kinds
    assert "".join(x[1]["text"] for x in events if x[0] == "token") == "Correct response"
    assert result["judge_passed"] is True
    assert openai.calls[0]["max_output_tokens"] == 800
    assert judge.calls == 1


def test_stream_rejected_answer_is_never_released():
    openai = FakeOpenAI("WRONG FIRST ANSWER")
    judge = FakeJudge([verdict(False), verdict(True)])
    with patch("doubtdesk.api.streaming.agenerate_general_answer", new_callable=AsyncMock, return_value="Fixed correct answer") as regenerate:
        events, result = collect(stream_answer({"question": "Question", "doc_ids": []}, openai, "openai-model", judge=judge))
    tokens = visible_text(events)
    assert tokens == "Fixed correct answer"
    assert "WRONG" not in tokens
    reset_at = next(index for index, event in enumerate(events) if event[0] == "reset")
    assert events[reset_at + 1] == ("status", {"stage": "revising"})
    assert result["answer"] == "Fixed correct answer"
    assert result["judge_attempts"] == 2
    assert judge.calls == 2
    assert regenerate.call_args.kwargs["feedback"] == "Repair the answer"


def test_stream_failed_judge_fails_closed_and_leaks_nothing():
    openai = FakeOpenAI("Potentially bad answer")
    judge = FakeJudge([verdict(False), verdict(False)])
    with patch("doubtdesk.api.streaming.agenerate_general_answer", new_callable=AsyncMock, return_value="Still invalid"):
        events, result = collect(stream_answer({"question": "Question", "doc_ids": []}, openai, "openai-model", judge=judge))
    tokens = visible_text(events)
    assert "Still invalid" not in tokens
    assert "Potentially bad answer" not in tokens
    assert result["judge_passed"] is False


def test_stream_low_pdf_specific_zero_llm_and_zero_judge_calls():
    openai = FakeOpenAI("Should never be generated")
    judge = FakeJudge([])
    prepared = {
        "question": "According to this PDF, what is the answer?",
        "doc_ids": ["docA"],
        "evidence_mode": "low",
        "retrieved_chunks": [{"doc_id": "docA", "page_number": 1, "text": "Irrelevant"}],
        "retrieved_pages": [],
    }
    events, result = collect(stream_answer(prepared, openai, "openai-model", judge=judge))
    assert result["skip_output_judge"]
    assert result["sources"] == []
    assert not openai.calls
    assert judge.calls == 0
    assert "couldn't find" in events[0][1]["text"]


def test_stream_low_non_pdf_question_uses_only_general_generation():
    openai = FakeOpenAI("A general fact")
    judge = FakeJudge([verdict(True)])
    prepared = {
        "question": "How does the solar system work?",
        "doc_ids": ["docA"],
        "evidence_mode": "low",
        "retrieved_chunks": [{"doc_id": "docA", "page_number": 1, "text": "Unrelated biology"}],
    }
    _, result = collect(stream_answer(prepared, openai, "openai-model", judge=judge))
    assert result["status"] == "general_knowledge"
    assert len(openai.calls) == 1
    assert judge.calls == 1


def test_stream_judge_http_503_yields_truthful_verification_failure():
    class UnavailableJudge:
        def evaluate(self, **kwargs):
            error = RuntimeError("Judge high demand")
            error.code = 503
            raise error

    openai = FakeOpenAI("Potential answer")
    events, result = collect(stream_answer(
        {"question": "What is gravity?", "doc_ids": []},
        openai, "openai-model", judge=UnavailableJudge(),
    ))
    tokens = visible_text(events)
    assert "Potential answer" not in tokens
    assert "verification is temporarily unavailable" in tokens
    assert any(event[0] == "reset" for event in events)
    assert result["judge_attempts"] == 1
    assert result["judge_failure_reason"] == "judge_unavailable"
    assert result["judge_needs_fallback"] is False


def test_stream_rejected_grounded_uses_vetted_general_fallback():
    import json
    grounded_json = json.dumps({"status": "sufficient", "answer": "UNVERIFIED PDF CLAIM", "sources": ["S1"]})
    openai = FakeOpenAI(grounded_json)
    judge = FakeJudge([verdict(False), verdict(False), verdict(True)])
    prepared = {
        "question": "Explain photosynthesis", "doc_ids": ["docA"],
        "evidence_mode": "text", "retrieved_pages": [],
        "retrieved_chunks": [{"doc_id":"docA", "page_number":1, "text":"Photosynthesis"}],
        "evidence_pages": [{"doc_id":"docA", "page_number":1}],
    }
    with (
        patch("doubtdesk.api.streaming.agenerate_vision_answer", new_callable=AsyncMock, return_value=SimpleNamespace(
            status="sufficient", answer="STILL UNVERIFIED", sources=[SimpleNamespace(
                doc_id="docA", page_numbers=[1], model_dump=lambda: {"doc_id":"docA", "page_numbers":[1]}
            )],
        )),
        patch("doubtdesk.api.streaming.astream_general", new=fake_general)
    ):
        events, result = collect(stream_answer(prepared, openai, "openai-model", judge=judge))
    tokens = visible_text(events)
    assert tokens == "Fallback explanation"
    assert result["status"] == "general_knowledge"
    assert result["judge_fallback_attempted"]
    assert "UNVERIFIED" not in tokens


async def fake_general(*args, **kwargs):
    yield ("result", {"status": "general_knowledge", "answer": "Fallback explanation", "sources": []})

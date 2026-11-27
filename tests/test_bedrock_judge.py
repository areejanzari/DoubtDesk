import asyncio
import json
import base64
from types import SimpleNamespace

import pytest

from doubtdesk.answering import output_guardrails
from doubtdesk.answering.output_guardrails import (
    BedrockOutputJudge, JudgeVerdict, build_output_judge, pacer_for,
)
from doubtdesk.config import Settings


class AsyncPacer:
    def __init__(self):
        self.calls = 0

    def acquire(self):
        self.calls += 1

    async def acquire_async(self):
        self.calls += 1


def verdict(**overrides):
    values = dict(passed=True, relevance=1, groundedness=1, completeness=1, safety="pass", feedback="")
    values.update(overrides)
    return JudgeVerdict(**values)


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def fake_client(*responses):
    return SimpleNamespace(messages=FakeMessages(responses))


def response(parsed, stop_reason="end_turn"):
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason,
                           usage=SimpleNamespace(input_tokens=120, output_tokens=30, cache_read_input_tokens=0))


GROUNDED = {"status": "grounded", "answer": "Water boils at 100 C [1].",
            "sources": [{"doc_id": "d", "page_numbers": [3]}]}


@pytest.fixture(autouse=True)
def no_usage_writes(monkeypatch):
    usage, failures = [], []
    monkeypatch.setattr(output_guardrails, "record_anthropic_usage", lambda **kw: usage.append(kw))
    monkeypatch.setattr(output_guardrails, "record_anthropic_failure", lambda **kw: failures.append(kw))
    return usage, failures


def test_bedrock_judge_sends_text_and_paraphrases_without_images(no_usage_writes):
    client = fake_client(response(verdict()))
    judge = BedrockOutputJudge(client, model_name="anthropic.claude-haiku-4-5", pacer=AsyncPacer())
    pages = [{"image_bytes": b"png-one"}, {"image_bytes": b"png-two"}, {"image_bytes": b"ignored"}]
    chunks = [{"doc_id": "d", "page_number": 3, "text": "Boiling point is 100 C."}]

    result = asyncio.run(judge.aevaluate(question="Boiling point?", result=GROUNDED, chunks=chunks,
                                         pages=pages, request_id="r1",
                                         paraphrases=["Boiling point?", "At what temperature does water boil?"]))

    assert result.passed
    call = client.messages.calls[0]
    assert call["model"] == "anthropic.claude-haiku-4-5"
    assert call["output_format"] is JudgeVerdict
    assert "temperature" not in call
    content = call["messages"][0]["content"]
    assert content[0]["type"] == "text" and "Boiling point is 100 C." in content[0]["text"]
    assert len(content) == 1
    payload = json.loads(content[0]["text"].split("\n\n", 1)[1])
    assert payload["question_paraphrases"] == ["At what temperature does water boil?"]
    assert no_usage_writes[0][0]["model"] == "anthropic.claude-haiku-4-5"


def test_bedrock_judge_general_answer_has_no_pdf_evidence():
    client = fake_client(response(verdict(groundedness=0.1)))
    judge = BedrockOutputJudge(client, model_name="m", pacer=AsyncPacer())
    general = {"status": "general_knowledge", "answer": "An answer.", "sources": []}

    result = judge.evaluate(question="Q?", result=general, chunks=[{"doc_id": "d", "page_number": 1, "text": "secret"}],
                            pages=[{"image_bytes": b"img"}])

    content = client.messages.calls[0]["messages"][0]["content"]
    assert result.passed
    assert len(content) == 1 and "secret" not in content[0]["text"]


def test_bedrock_judge_enforces_thresholds():
    judge = BedrockOutputJudge(fake_client(response(verdict(groundedness=0.5))), model_name="m", pacer=AsyncPacer())
    assert not asyncio.run(judge.aevaluate(question="Q?", result=GROUNDED)).passed


@pytest.mark.parametrize("stop_reason", ["max_tokens", "refusal"])
def test_bedrock_judge_rejects_truncated_or_refused_output(stop_reason, no_usage_writes):
    judge = BedrockOutputJudge(fake_client(response(None, stop_reason)), model_name="m", pacer=AsyncPacer())
    with pytest.raises(RuntimeError):
        asyncio.run(judge.aevaluate(question="Q?", result=GROUNDED, request_id="r"))
    assert no_usage_writes[1]


def test_bedrock_judge_uses_backup_model_on_overload(monkeypatch):
    monkeypatch.setenv("OUTPUT_JUDGE_FALLBACK_MODEL", "backup-model")
    overloaded = RuntimeError("overloaded")
    overloaded.status_code = 529
    client = fake_client(overloaded, response(verdict()))
    pacer = AsyncPacer()
    judge = BedrockOutputJudge(client, model_name="primary", pacer=pacer)

    assert asyncio.run(judge.aevaluate(question="Q?", result=GROUNDED)).passed
    assert [call["model"] for call in client.messages.calls] == ["primary", "backup-model"]
    assert pacer.calls == 2


def test_pacers_are_separate_per_model():
    assert pacer_for("model-a") is pacer_for("model-a")
    assert pacer_for("model-a") is not pacer_for("model-b")


def test_build_output_judge_requires_region_for_bedrock(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    config = Settings(_env_file=None, output_judge_enabled=True, output_judge_provider="bedrock", aws_region="")
    with pytest.raises(RuntimeError, match="AWS_REGION"):
        build_output_judge(config)


def test_build_output_judge_disabled():
    assert build_output_judge(Settings(_env_file=None, output_judge_enabled=False)) == (None, None)


def test_build_output_judge_bedrock(monkeypatch):
    created = {}

    class FakeMantle:
        def __init__(self, **kwargs):
            created.update(kwargs)

    import anthropic
    monkeypatch.setattr(anthropic, "AsyncAnthropicBedrockMantle", FakeMantle)
    config = Settings(_env_file=None, output_judge_enabled=True, output_judge_provider="bedrock", aws_region="us-west-2",
                      output_judge_model="anthropic.claude-haiku-4-5")
    client, judge = build_output_judge(config)
    assert isinstance(client, FakeMantle) and created["aws_region"] == "us-west-2"
    assert isinstance(judge, BedrockOutputJudge) and judge.model_name == "anthropic.claude-haiku-4-5"


class FakeCompletions:
    def __init__(self, content, finish_reason="stop", error=None):
        self.content = content
        self.finish_reason = finish_reason
        self.error = error
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        usage = SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=150)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)], usage=usage)


def openai_judge(completions):
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return output_guardrails.BedrockOpenAIOutputJudge(client, model_name="qwen.qwen3-vl-235b-a22b-instruct", pacer=AsyncPacer())


def grounded_result():
    return {"status": "grounded", "answer": "Birds are uricotelic [1].", "sources": []}


def test_openai_judge_sends_text_only_and_parses_wrapped_json(monkeypatch):
    rows = []
    monkeypatch.setattr(output_guardrails, "record_anthropic_usage", lambda **kwargs: rows.append(kwargs))
    completions = FakeCompletions('Here you go:\n```json\n' + verdict().model_dump_json() + '\n```')
    judge = openai_judge(completions)
    pages = [{"image_bytes": b"png-bytes"}]
    result = asyncio.run(judge.aevaluate(question="Which animals are uricotelic?", result=grounded_result(),
                                         chunks=[{"doc_id": "d", "page_number": 1, "text": "Birds are uricotelic."}],
                                         pages=pages, request_id="r1", paraphrases=["Name uricotelic animals"]))
    assert result.passed
    call = completions.calls[0]
    assert call["model"] == "qwen.qwen3-vl-235b-a22b-instruct" and call["temperature"] == 0
    content = call["messages"][0]["content"]
    assert "matches this schema" in content[0]["text"]
    assert len(content) == 1 and "Name uricotelic animals" in content[0]["text"]
    assert rows[0]["provider"] == "bedrock_openai"


def test_openai_judge_strips_think_blocks():
    completions = FakeCompletions("<think>{\"passed\": true}</think>" + verdict(passed=False, groundedness=0.2).model_dump_json())
    result = asyncio.run(openai_judge(completions).aevaluate(question="q", result=grounded_result()))
    assert not result.passed and result.groundedness == 0.2


@pytest.mark.parametrize("content, finish", [("no json here", "stop"), ('{"passed": true', "length")])
def test_openai_judge_rejects_unusable_output(content, finish):
    with pytest.raises(Exception):
        asyncio.run(openai_judge(FakeCompletions(content, finish)).aevaluate(question="q", result=grounded_result()))


def test_openai_judge_uses_backup_model_on_rate_limit(monkeypatch):
    monkeypatch.setattr(output_guardrails, "runtime_settings",
                        lambda: Settings(_env_file=None, output_judge_fallback_model="mistral.ministral-3-14b-instruct"))
    error = type("RateLimited", (Exception,), {"status_code": 429})()

    class Flaky(FakeCompletions):
        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise error
            return await FakeCompletions.create(FakeCompletions(verdict().model_dump_json()), **kwargs)

    completions = Flaky("")
    asyncio.run(openai_judge(completions).aevaluate(question="q", result=grounded_result()))
    assert [call["model"] for call in completions.calls] == ["qwen.qwen3-vl-235b-a22b-instruct", "mistral.ministral-3-14b-instruct"]


def test_build_output_judge_bedrock_openai(monkeypatch):
    created = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            created.update(kwargs)

    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", FakeOpenAI)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    config = Settings(_env_file=None, output_judge_enabled=True, aws_region="ap-southeast-2")
    client, judge = build_output_judge(config)
    assert created["base_url"] == "https://bedrock-mantle.ap-southeast-2.api.aws/v1" and created["api_key"] == "token"
    assert isinstance(judge, output_guardrails.BedrockOpenAIOutputJudge)
    assert judge.model_name == "qwen.qwen3-vl-235b-a22b-instruct"


def test_build_output_judge_bedrock_openai_requires_token(monkeypatch):
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    monkeypatch.delenv("ANTHROPIC_AWS_API_KEY", raising=False)
    config = Settings(_env_file=None, output_judge_enabled=True, aws_region="ap-southeast-2")
    with pytest.raises(RuntimeError, match="AWS_BEARER_TOKEN_BEDROCK"):
        build_output_judge(config)


def test_openai_judge_works_with_real_async_openai_client():
    import httpx
    import openai

    seen = []

    def handler(request):
        seen.append(request)
        body = {
            "id": "c1", "object": "chat.completion", "created": 0, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": verdict().model_dump_json()}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return httpx.Response(200, json=body)

    async def run():
        client = openai.AsyncOpenAI(base_url="https://bedrock-mantle.ap-southeast-2.api.aws/v1", api_key="token",
                                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
        judge = output_guardrails.BedrockOpenAIOutputJudge(client, model_name="qwen", pacer=AsyncPacer())
        try:
            return await judge.aevaluate(question="q", result=grounded_result())
        finally:
            await client.close()

    assert asyncio.run(run()).passed
    assert seen[0].url.path == "/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer token"


def test_review_loop_passes_paraphrases_to_judge():
    captured = {}

    class CapturingJudge(output_guardrails.OutputJudge):
        async def aevaluate(self, **kwargs):
            captured.update(kwargs)
            return verdict()

    judge = CapturingJudge(None, model_name="m", pacer=AsyncPacer())
    result = {"status": "general_knowledge", "answer": "An answer.", "sources": []}

    async def regenerate(feedback):
        return result

    asyncio.run(output_guardrails.areview_and_repair(result=dict(result), question="Q?", judge=judge,
                                                     regenerate=regenerate, paraphrases=["Q?", "Other wording?"]))
    assert captured["paraphrases"] == ["Q?", "Other wording?"]
    assert "pages" in captured


def test_judge_sees_only_excerpts_from_cited_pages():
    chunks = [{"doc_id": "d", "page_number": page, "text": f"text of page {page}"} for page in (1, 2, 3, 4)]
    result = {"status": "grounded", "answer": "A [1] [2].",
              "sources": [{"doc_id": "d", "page_numbers": [2, 4]}]}
    text, grounded = output_guardrails.judge_inputs(question="q", result=result, chunks=chunks)
    excerpts = json.loads(text.split("\n\n", 1)[1])["retrieved_pdf_text"]
    assert grounded and [item["page_number"] for item in excerpts] == [2, 4]


def test_judge_falls_back_to_all_excerpts_when_citations_match_nothing():
    chunks = [{"doc_id": "d", "page_number": page, "text": f"text of page {page}"} for page in (1, 2)]
    for sources in ([], [{"doc_id": "other", "page_numbers": [9]}]):
        result = {"status": "grounded", "answer": "A.", "sources": sources}
        text, _ = output_guardrails.judge_inputs(question="q", result=result, chunks=chunks)
        assert [item["page_number"] for item in json.loads(text.split("\n\n", 1)[1])["retrieved_pdf_text"]] == [1, 2]

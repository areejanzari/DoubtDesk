import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest

from doubtdesk.config import Settings
from doubtdesk.guardrails import paraphrasing
from doubtdesk.guardrails.paraphrasing import (
    BedrockChat, ContextualParaphraseResult, aresolve_and_paraphrase, build_paraphraser, resolve_and_paraphrase,
)

REPLY = {"resolved_question": "Where is Bowman's capsule located?",
         "questions": ["Where is Bowman's capsule found?", "What is the location of Bowman's capsule?",
                       "Bowman's capsule position in the nephron"]}


def mock_chat(content, finish_reason="stop", seen=None):
    def handler(request):
        if seen is not None:
            seen.append(request)
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0, "model": "qwen",
            "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 60, "total_tokens": 260},
        })
    client = openai.AsyncOpenAI(base_url="https://bedrock-mantle.ap-south-1.api.aws/v1", api_key="token",
                                http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return BedrockChat(client)


def run(chat, **kwargs):
    async def go():
        try:
            return await aresolve_and_paraphrase(client=chat, model_name="qwen", **kwargs)
        finally:
            await chat.close()
    return asyncio.run(go())


def test_bedrock_paraphrase_sends_history_and_parses_reply(monkeypatch):
    rows = []
    monkeypatch.setattr("doubtdesk.analytics.tracker.record_anthropic_usage", lambda **kwargs: rows.append(kwargs))
    seen = []
    history = [{"role": "user", "content": "What is Bowman's capsule?"}]
    result = run(mock_chat("```json\n" + json.dumps(REPLY) + "\n```", seen=seen),
                 question="where is it located?", history=history, request_id="r1")
    assert result.resolved_question == REPLY["resolved_question"] and len(result.questions) == 3
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/chat/completions" and body["model"] == "qwen" and body["max_tokens"] == 650
    assert "matches this schema" in body["messages"][0]["content"]
    assert json.loads(body["messages"][1]["content"]) == {"history": history, "current_question": "where is it located?"}
    assert rows[0]["provider"] == "bedrock_openai" and rows[0]["operation"] == "paraphrasing"


@pytest.mark.parametrize("content, finish", [
    ("no json", "stop"),
    (json.dumps({"resolved_question": "q", "questions": ["only one"]}), "stop"),
    ('{"resolved_question": "q"', "length"),
])
def test_bedrock_paraphrase_rejects_unusable_replies(monkeypatch, content, finish):
    failures = []
    monkeypatch.setattr("doubtdesk.analytics.tracker.record_anthropic_failure", lambda **kwargs: failures.append(kwargs))
    with pytest.raises(Exception):
        run(mock_chat(content, finish), question="q", history=[], request_id="r1")
    assert failures and failures[0]["provider"] == "bedrock_openai"


def test_sync_paraphrase_supports_bedrock_chat():
    chat = mock_chat(json.dumps(REPLY))
    try:
        result = resolve_and_paraphrase(chat, "qwen", "where is it located?", [])
    finally:
        asyncio.run(chat.close())
    assert isinstance(result, ContextualParaphraseResult)


def test_build_paraphraser_defaults_to_openai_client(monkeypatch):
    monkeypatch.delenv("PARAPHRASE_PROVIDER", raising=False)
    monkeypatch.delenv("PARAPHRASE_MODEL", raising=False)
    llm = object()
    client, model = build_paraphraser(Settings(_env_file=None), llm, "gpt-6-luna")
    assert client is llm and model == "gpt-6-luna"
    client, model = build_paraphraser(Settings(_env_file=None, paraphrase_model="other"), llm, "gpt-6-luna")
    assert model == "other"


def test_build_paraphraser_bedrock(monkeypatch):
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "token")
    config = Settings(_env_file=None, paraphrase_provider="bedrock_openai", aws_region="ap-south-1")
    client, model = build_paraphraser(config, object(), "gpt-6-luna")
    assert isinstance(client, BedrockChat) and model == paraphrasing.DEFAULT_BEDROCK_PARAPHRASE_MODEL
    assert str(client.client.base_url) == "https://bedrock-mantle.ap-south-1.api.aws/v1/"
    asyncio.run(client.close())


def test_build_paraphraser_bedrock_requires_region(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
    config = Settings(_env_file=None, paraphrase_provider="bedrock_openai", aws_region="")
    with pytest.raises(RuntimeError, match="AWS_REGION is required when PARAPHRASE_PROVIDER"):
        build_paraphraser(config, object(), "m")


def test_preload_sdk_resources_touches_lazy_clients_and_skips_others():
    from functools import cached_property
    from doubtdesk.llm import preload_sdk_resources

    touched = []

    class Completions:
        pass

    class Chat:
        @cached_property
        def completions(self):
            touched.append("chat.completions")
            return Completions()

    class FakeOpenAI:
        @cached_property
        def chat(self):
            touched.append("chat")
            return Chat()

        @cached_property
        def responses(self):
            touched.append("responses")
            return object()

    class Unrelated:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected attribute {name}")

    preload_sdk_resources(FakeOpenAI(), None, Unrelated())
    assert touched == ["chat", "chat.completions", "responses"]

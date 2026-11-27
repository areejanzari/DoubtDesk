import asyncio
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from doubtdesk.answering.output_guardrails import JudgeVerdict, areview_and_repair, review_and_repair
from doubtdesk.guardrails.paraphrasing import ContextualParaphraseResult
from doubtdesk.llm import acall, iterate_with_result, response_stream, run_coroutine
from doubtdesk.rate_limit import RequestPacer


def test_ahybrid_search_runs_dense_and_lexical_concurrently():
    from doubtdesk.retrieval import hybrid
    barrier = threading.Barrier(2, timeout=5)
    dense = [{"chunk_id": "a", "doc_id": "d", "page_number": 1, "text": "x", "distance": 0.1}]
    sparse = [{"chunk_id": "b", "doc_id": "d", "page_number": 2, "text": "y"}]

    def search_chunks(embeddings, doc_ids, top_k):
        barrier.wait()
        return [dense]

    def lexical_candidates(questions, doc_ids, top_k):
        barrier.wait()
        return [sparse]

    with (
        patch.object(hybrid, "embed_queries", return_value=[[0.0]]),
        patch.object(hybrid, "search_chunks", side_effect=search_chunks),
        patch.object(hybrid, "lexical_candidates", side_effect=lexical_candidates),
    ):
        concurrent = asyncio.run(hybrid.ahybrid_search(Mock(), ["q"], ["d"], top_k=5))
    with (
        patch.object(hybrid, "embed_queries", return_value=[[0.0]]),
        patch.object(hybrid, "search_chunks", return_value=[dense]),
        patch.object(hybrid, "lexical_candidates", return_value=[sparse]),
    ):
        sequential = hybrid.hybrid_search(Mock(), ["q"], ["d"], top_k=5)
    assert concurrent == sequential
    assert [item["chunk_id"] for item in concurrent] == ["a", "b"]


def test_ahybrid_search_validation_matches_sync():
    from doubtdesk.retrieval.hybrid import ahybrid_search
    with pytest.raises(ValueError):
        asyncio.run(ahybrid_search(Mock(), ["q"], ["d"], top_k=0))
    assert asyncio.run(ahybrid_search(Mock(), [], ["d"])) == []


def exchange(item_id, docs=("bio",)):
    return {"id": item_id, "question": f"q{item_id}", "answer": "a" * 2000, "doc_ids": list(docs)}


@pytest.mark.parametrize("doc_ids", [["bio"], []])
def test_abuild_history_matches_sync_history(doc_ids):
    from doubtdesk.memory import context
    recent = [exchange(9), exchange(8)]
    ranked = [9, 8, 4, 3, 2, 1]

    def search(*, limit, exclude_ids, **kwargs):
        hits = [{"exchange_id": item} for item in ranked if item not in exclude_ids]
        return hits[:limit]

    def by_ids(conversation_id, user_id, ids):
        return [exchange(item, docs=("chem",) if item == 3 else ("bio",)) for item in ids]

    with (
        patch.object(context, "get_user_recent_exchanges", return_value=recent),
        patch.object(context, "get_user_eligible_exchanges_by_ids", side_effect=by_ids),
    ):
        expected = context.build_history("chat", "alice", "q", doc_ids, model=Mock(), search=search)
        actual = asyncio.run(context.abuild_history("chat", "alice", "q", doc_ids, model=Mock(), search=search))
    assert actual == expected
    assert [item["question"] for item in actual][:3] == (["q4", "q2", "q9"] if doc_ids else ["q4", "q3", "q2"])


def test_async_pacer_sleeps_without_blocking_event_loop():
    pacer = RequestPacer(600)
    ticks = []

    async def ticker(stop):
        while not stop.is_set():
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    async def main():
        stop = asyncio.Event()
        task = asyncio.create_task(ticker(stop))
        started = time.monotonic()
        await pacer.acquire_async()
        await pacer.acquire_async()
        elapsed = time.monotonic() - started
        stop.set()
        await task
        return elapsed

    elapsed = asyncio.run(main())
    assert elapsed >= 0.09
    assert len(ticks) >= 5


def verdict(passed, feedback="Repair"):
    return JudgeVerdict(passed=passed, relevance=1 if passed else 0.2, groundedness=1, completeness=1, safety="pass", feedback=feedback)


class SequenceJudge:
    def __init__(self, items):
        self.items = list(items)

    def evaluate(self, **kwargs):
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def unavailable():
    error = RuntimeError("busy")
    error.code = 503
    return error


@pytest.mark.parametrize("judgements,regenerated", [
    ([verdict(False), verdict(True)], "Fixed"),
    ([verdict(False), verdict(False)], "Still wrong"),
    ([unavailable()], "unused"),
    ([verdict(False)], ValueError("bad candidate")),
])
def test_async_review_matches_sync_review(judgements, regenerated):
    result = {"status": "general_knowledge", "answer": "Draft", "sources": []}

    def regenerate(feedback):
        if isinstance(regenerated, Exception):
            raise regenerated
        return {"status": "general_knowledge", "answer": regenerated, "sources": []}

    async def aregenerate(feedback):
        return regenerate(feedback)

    expected = review_and_repair(result=dict(result), question="Q", judge=SequenceJudge(judgements), regenerate=regenerate)
    actual = asyncio.run(areview_and_repair(result=dict(result), question="Q", judge=SequenceJudge(judgements), regenerate=aregenerate))
    assert actual == expected


def test_async_review_prefers_native_async_judge():
    class AsyncJudge:
        calls = 0

        def evaluate(self, **kwargs):
            raise AssertionError("sync judge used")

        async def aevaluate(self, **kwargs):
            AsyncJudge.calls += 1
            return verdict(True)

    async def regenerate(feedback):
        raise AssertionError("not needed")

    result = asyncio.run(areview_and_repair(
        result={"status": "general_knowledge", "answer": "Draft", "sources": []},
        question="Q", judge=AsyncJudge(), regenerate=regenerate,
    ))
    assert result["judge_passed"] is True and AsyncJudge.calls == 1


class AsyncStream:
    def __init__(self, pieces):
        self.pieces = pieces
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True
        return False

    async def _events(self):
        for piece in self.pieces:
            await asyncio.sleep(0)
            yield SimpleNamespace(type="response.output_text.delta", delta=piece)

    def __aiter__(self):
        return self._events()

    async def get_final_response(self):
        return SimpleNamespace(usage=None, status="completed")


class AsyncOnlyClient:
    def __init__(self):
        self.streams = []
        self.parsed = []
        self.responses = self

    async def parse(self, *, model, input, text_format, **kwargs):
        self.parsed.append(text_format)
        question = json.loads(input[1]["content"])["current_question"]
        return SimpleNamespace(output_parsed={"resolved_question": question, "questions": [question, f"Explain {question}", f"Define {question}"]},
                               usage=None, output_text="")

    def stream(self, **kwargs):
        stream = AsyncStream(["General ", "answer."])
        self.streams.append(stream)
        return stream


def test_prepared_graph_ainvoke_uses_native_async_client():
    from doubtdesk.api.streaming import PreparedAnswerGraph
    from doubtdesk.graph.builder import build_preparation_graph
    classifier = Mock()
    classifier.predict.return_value = [{"question": "q", "label": 0, "off_topic_probability": 0.0}] * 3
    client = AsyncOnlyClient()
    preparation = build_preparation_graph(embedding_model=Mock(), client=client, model_name="m", scope_classifier=classifier)
    graph = PreparedAnswerGraph(preparation, client, "m", None)
    result = asyncio.run(graph.ainvoke({"user_id": "alice", "question": "What is gravity?", "doc_ids": []}))
    assert result["status"] == "general_knowledge"
    assert result["answer"] == "General answer."
    assert result["paraphrases"][1] == "Explain What is gravity?"
    assert client.parsed == [ContextualParaphraseResult]
    assert client.streams[0].closed


def test_sync_stream_bridge_uses_async_client_and_returns_result():
    from doubtdesk.api.streaming import stream_general
    client = AsyncOnlyClient()
    generator = stream_general(client, "m", "Gravity?")
    events = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as finished:
        result = finished.value
    assert [data["text"] for _, data in events] == ["General ", "answer."]
    assert result == {"status": "general_knowledge", "answer": "General answer.", "sources": []}


def test_response_stream_runs_sync_streams_in_worker_thread():
    main_thread = threading.get_ident()
    seen = []

    class SyncStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            seen.append("exit")
            return False

        def __iter__(self):
            seen.append(threading.get_ident())
            yield SimpleNamespace(type="response.output_text.delta", delta="x")

        def get_final_response(self):
            return "final"

    client = SimpleNamespace(responses=SimpleNamespace(stream=lambda **kwargs: SyncStream()))

    async def main():
        async with response_stream(client, model="m") as stream:
            items = [event.delta async for event in stream]
            final = await stream.get_final_response()
        return items, final

    assert asyncio.run(main()) == (["x"], "final")
    assert seen[0] != main_thread and seen[-1] == "exit"


def test_bridge_shares_one_context_across_steps_and_closes_generator():
    import contextvars
    marker = contextvars.ContextVar("marker", default=None)
    closed = []

    async def agen():
        token = marker.set("inside")
        try:
            yield ("token", 1)
            assert marker.get() == "inside"
            yield ("token", 2)
            yield ("result", "done")
        finally:
            marker.reset(token)
            closed.append(True)

    generator = iterate_with_result(agen())
    assert next(generator) == ("token", 1)
    generator.close()
    assert closed == [True]
    assert marker.get() is None
    assert run_coroutine(acall(lambda: 5)) == 5


class AsyncCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.models = []

    async def create(self, *, model, messages, **kwargs):
        self.models.append(model)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class AsyncPacer:
    def __init__(self):
        self.calls = 0

    def acquire(self):
        raise AssertionError("blocking pacer used")

    async def acquire_async(self):
        self.calls += 1


def chat_reply(content):
    message = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)], usage=None)


def bedrock_client(outcomes):
    completions = AsyncCompletions(outcomes)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_bedrock_aevaluate_uses_async_client_pacer_and_records_usage(monkeypatch):
    from doubtdesk.answering import output_guardrails
    from doubtdesk.answering.output_guardrails import BedrockOpenAIOutputJudge
    usage, failures = Mock(), Mock()
    monkeypatch.setattr(output_guardrails, "record_anthropic_usage", usage)
    monkeypatch.setattr(output_guardrails, "record_anthropic_failure", failures)
    client, completions = bedrock_client([chat_reply(verdict(True).model_dump_json())])
    pacer = AsyncPacer()
    judge = BedrockOpenAIOutputJudge(client, model_name="primary", pacer=pacer)
    result = asyncio.run(judge.aevaluate(
        question="What is ATP?",
        result={"status": "grounded", "answer": "ATP stores energy", "sources": [{"doc_id": "d", "page_numbers": [1]}]},
        chunks=[{"doc_id": "d", "page_number": 1, "text": "ATP"}], request_id="req",
    ))
    assert result.passed and completions.models == ["primary"] and pacer.calls == 1
    assert usage.call_args.kwargs["request_id"] == "req"
    failures.assert_not_called()


def test_bedrock_aevaluate_fails_over_to_backup_model(monkeypatch):
    from doubtdesk.answering import output_guardrails
    from doubtdesk.answering.output_guardrails import BedrockOpenAIOutputJudge
    monkeypatch.setenv("OUTPUT_JUDGE_FALLBACK_MODEL", "backup")
    failures = Mock()
    monkeypatch.setattr(output_guardrails, "record_anthropic_failure", failures)
    monkeypatch.setattr(output_guardrails, "record_anthropic_usage", Mock())
    client, completions = bedrock_client([unavailable(), chat_reply(json.dumps(verdict(True).model_dump()))])
    judge = BedrockOpenAIOutputJudge(client, model_name="primary", pacer=AsyncPacer())
    result = asyncio.run(judge.aevaluate(question="Q", result={"status": "general_knowledge", "answer": "A", "sources": []},
                                         request_id="req"))
    assert result.passed and completions.models == ["primary", "backup"]
    assert failures.call_args.kwargs["error_type"] == "RuntimeError"


def test_bedrock_aevaluate_without_backup_reraises(monkeypatch):
    from doubtdesk.answering.output_guardrails import BedrockOpenAIOutputJudge
    monkeypatch.delenv("OUTPUT_JUDGE_FALLBACK_MODEL", raising=False)
    client, _ = bedrock_client([unavailable()])
    judge = BedrockOpenAIOutputJudge(client, model_name="primary", pacer=AsyncPacer())
    with pytest.raises(RuntimeError):
        asyncio.run(judge.aevaluate(question="Q", result={"status": "general_knowledge", "answer": "A", "sources": []}))

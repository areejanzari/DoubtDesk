import json
import sys
import time
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import anyio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from doubtdesk.api.stream_context import ContextStreamingResponse
from doubtdesk.observability.tracing import trace_generation, trace_run, trace_stage, observe_llm_response


@pytest.fixture
def trace_capture(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("doubtdesk-regression")
    data = {}
    spans = []

    @contextmanager
    def start_as_current_observation(**kwargs):
        with tracer.start_as_current_span(kwargs["name"]) as span:
            span_id = span.get_span_context().span_id
            data[span_id] = dict(kwargs)
            spans.append(span)
            yield SimpleNamespace(update=lambda **updates: data[span_id].update(updates))

    @contextmanager
    def propagate_attributes(**kwargs):
        token = otel_context.attach(otel_context.set_value("test-user", kwargs.get("user_id")))
        try:
            yield
        finally:
            otel_context.detach(token)

    langfuse = ModuleType("langfuse")
    langfuse.get_client = lambda: SimpleNamespace(start_as_current_observation=start_as_current_observation)
    langfuse.propagate_attributes = propagate_attributes
    langchain = ModuleType("langfuse.langchain")
    langchain.CallbackHandler = lambda: object()
    monkeypatch.setitem(sys.modules, "langfuse", langfuse)
    monkeypatch.setitem(sys.modules, "langfuse.langchain", langchain)
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "test-public")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "test-secret")
    yield SimpleNamespace(exporter=exporter, data=data, spans=spans)
    provider.shutdown()


def finished(capture):
    return capture.exporter.get_finished_spans()


def assert_single_tree(capture, root_name, *, allow_upstream_parent=False):
    spans = finished(capture)
    roots = [span for span in spans if span.name == root_name]
    assert len(roots) == 1
    root = roots[0]
    assert {span.context.trace_id for span in spans} == {root.context.trace_id}
    if root.parent is not None:
        assert allow_upstream_parent, "Unexpected upstream parent in isolated trace"
        assert root.parent.is_valid
        assert root.parent.trace_id == root.context.trace_id
        assert root.parent.span_id not in {span.context.span_id for span in spans}
    else:
        assert root.parent is None
    by_id = {span.context.span_id: span for span in spans}
    for span in spans:
        visited = set()
        while span.context.span_id != root.context.span_id:
            assert span.context.span_id not in visited, "Cycle in trace hierarchy"
            visited.add(span.context.span_id)
            assert span.parent is not None, "Orphaned request observation"
            assert span.parent.span_id in by_id, "Observation is outside the request subtree"
            span = by_id[span.parent.span_id]
    return root, {span.name: span for span in spans}


def test_stream_keeps_context_across_yields(trace_capture, caplog):
    ids = []

    def content():
        with trace_run("turn", user_id="alice", session_id="session"):
            for index in range(3):
                ids.append(otel_trace.get_current_span().get_span_context().span_id)
                assert otel_context.get_value("test-user") == "alice"
                with trace_stage(f"part-{index}"):
                    yield str(index)
            assert otel_context.get_value("test-user") == "alice"

    async def run():
        response = ContextStreamingResponse(content())
        parts = []
        async def send(message):
            if message["type"] == "http.response.body":
                parts.append(message["body"])
        await response.stream_response(send)
        assert b"".join(parts) == b"012"
        assert not otel_trace.get_current_span().get_span_context().is_valid
        assert otel_context.get_value("test-user") is None

    anyio.run(run)
    root, spans = assert_single_tree(trace_capture, "turn")
    assert ids == [root.context.span_id] * 3
    assert all(spans[f"part-{i}"].parent.span_id == root.context.span_id for i in range(3))
    assert "Failed to detach context" not in caplog.text


def test_disconnect_closes_nested_stream_and_trace(trace_capture, caplog):
    cleaned = []

    def content():
        with trace_run("turn"):
            with trace_generation("general_answer", "fake"):
                try:
                    yield "token"
                finally:
                    cleaned.append(True)

    async def run():
        response = ContextStreamingResponse(content())
        async def send(message):
            if message["type"] == "http.response.body":
                raise OSError("connection closed")
        with pytest.raises(OSError, match="connection closed"):
            await response.stream_response(send)
        assert response.context_iterator.closed
        assert not otel_trace.get_current_span().get_span_context().is_valid

    anyio.run(run)
    assert cleaned == [True]
    root, spans = assert_single_tree(trace_capture, "turn")
    assert spans["general_answer"].parent.span_id == root.context.span_id
    assert "Failed to detach context" not in caplog.text


def test_two_interleaved_streams_keep_separate_traces(trace_capture, caplog):
    def content(label):
        with trace_run(label, user_id=label):
            for i in range(2):
                assert otel_context.get_value("test-user") == label
                with trace_stage(f"{label}-{i}"):
                    yield label

    async def run():
        async def consume(label):
            response = ContextStreamingResponse(content(label))
            async def send(message):
                await anyio.sleep(0.001)
            await response.stream_response(send)
        async with anyio.create_task_group() as group:
            group.start_soon(consume, "alice")
            group.start_soon(consume, "bob")
    anyio.run(run)
    spans = {span.name: span for span in finished(trace_capture)}
    assert spans["alice"].context.trace_id != spans["bob"].context.trace_id
    for label in ("alice", "bob"):
        for i in range(2):
            assert spans[f"{label}-{i}"].parent.span_id == spans[label].context.span_id
    assert "Failed to detach context" not in caplog.text


def test_generation_updates_existing_span_not_duplicate(trace_capture, monkeypatch):
    from doubtdesk.analytics import tracker
    record = Mock()
    monkeypatch.setattr(tracker, "record_openai_usage", record)
    response = SimpleNamespace(output_text="answer", usage=SimpleNamespace(input_tokens=7, output_tokens=3))
    with trace_run("turn"):
        with trace_generation("general_answer", "fake", "req"):
            time.sleep(0.01)
            observe_llm_response("general_answer", response, "fake", "req", "chat")
    root, spans = assert_single_tree(trace_capture, "turn")
    generation = spans["general_answer"]
    assert len(finished(trace_capture)) == 2
    assert generation.parent.span_id == root.context.span_id
    assert generation.end_time - generation.start_time >= 10_000_000
    assert trace_capture.data[generation.context.span_id]["output"] == "answer"
    assert trace_capture.data[generation.context.span_id]["usage_details"] == {"input": 7, "output": 3}
    record.assert_called_once()
    assert record.call_args.kwargs["latency_ms"] >= 10


class FakeStream:
    def __init__(self, raw):
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        for pos in range(0, len(self.raw), 8):
            time.sleep(0.001)
            yield SimpleNamespace(type="response.output_text.delta", delta=self.raw[pos:pos + 8])

    def get_final_response(self):
        return SimpleNamespace(output_text=self.raw, usage=SimpleNamespace(input_tokens=30, output_tokens=15))


@pytest.mark.parametrize("mode,expected", [("general", "general_answer"), ("text", "grounded_text"), ("visual", "grounded_visual"), ("unavailable", "general_answer")])
def test_real_ask_stream_route_records_one_complete_turn(mode, expected, tmp_path, monkeypatch, trace_capture, caplog):
    import importlib
    routes = importlib.import_module("doubtdesk.api.routes")
    streaming = importlib.import_module("doubtdesk.api.streaming")
    from doubtdesk.answering.output_guardrails import JudgeVerdict
    from doubtdesk.analytics import tracker
    from doubtdesk.auth.dependencies import require_current_user

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(routes, "validate_user_document_ids", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "claim_or_check_conversation", lambda **kw: True)
    save = Mock()
    monkeypatch.setattr(routes, "_save_exchange_and_memory", save)
    monkeypatch.setattr(streaming, "get_document_names", lambda *a, **kw: {"doc": "biology.pdf"})
    monkeypatch.setattr(tracker, "record_openai_usage", Mock())

    answer = "Plants absorb carbon dioxide."
    grounded = mode in ("text", "visual")
    sources = [{"doc_id": "doc", "page_numbers": [1]}] if grounded else []
    raw = json.dumps({"status": "sufficient", "answer": answer, "sources": ["S1"]}) if grounded else answer
    provider = Mock()
    provider.responses.stream.side_effect = lambda **kwargs: FakeStream(raw)
    prepared = {"question": "How do plants get carbon dioxide?", "doc_ids": ["11111111-1111-4111-8111-111111111111"] if grounded else [],
                "retrieved_chunks": [{"doc_id": "doc", "page_number": 1, "text": answer}] if grounded else [],
                "retrieved_pages": [], "evidence_pages": [{"doc_id": "doc", "page_number": 1}] if grounded else [],
                "evidence_mode": mode}
    if mode == "visual":
        image = tmp_path / "page.png"
        image.write_bytes(b"fake image")
        prepared["retrieved_pages"] = [{"doc_id": "doc", "page_number": 1, "image_path": str(image)}]

    def prepare(*args, **kwargs):
        with trace_stage("LangGraph", as_type="chain"):
            return prepared

    judge = Mock()
    judge.model_name = "fake-judge"
    if mode == "unavailable":
        error = RuntimeError("provider down")
        error.code = 503
        judge.evaluate.side_effect = error
    else:
        judge.evaluate.return_value = JudgeVerdict(passed=True, relevance=1, groundedness=1, completeness=1, safety="pass")
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_current_user] = lambda: {"user_id": "alice"}
    app.state.preparation_graph = SimpleNamespace(invoke=prepare)
    app.state.llm_client = provider
    app.state.model_name = "fake-model"
    app.state.output_judge = judge
    with TestClient(app) as client:
        result = client.post("/ask/stream", json={"question": prepared["question"], "doc_ids": prepared["doc_ids"], "conversation_id": str(uuid4())})
    assert result.status_code == 200
    event = next(block for block in result.text.split("\n\n") if block.startswith("event: complete\n"))
    delivered = json.loads(event.split("data: ", 1)[1])
    root, spans = assert_single_tree(trace_capture, "doubtdesk.ask.stream", allow_upstream_parent=True)
    assert {"LangGraph", expected, "output_verification", "save_exchange"}.issubset(spans)
    for name in ("LangGraph", expected, "output_verification", "save_exchange"):
        assert spans[name].parent.span_id == root.context.span_id
    assert trace_capture.data[root.context.span_id]["output"]["answer"] == delivered["answer"]
    assert trace_capture.data[root.context.span_id]["output"]["status"] == delivered["status"]
    assert trace_capture.data[spans[expected].context.span_id]["output"] == raw
    assert spans[expected].end_time > spans[expected].start_time
    assert sum(span.name == expected for span in finished(trace_capture)) == 1
    if mode == "unavailable":
        assert delivered["status"] == "verification_failed"
        assert answer not in result.text
        assert trace_capture.data[spans["output_verification"].context.span_id]["level"] == "ERROR"
    else:
        assert delivered["answer"] == answer
        assert delivered["sources"] == sources
    assert save.call_count == 1
    assert provider.responses.stream.call_count == 1
    assert "Failed to detach context" not in caplog.text


def test_regeneration_is_on_same_trace_without_duplicate_accounting(trace_capture, monkeypatch):
    from doubtdesk.api.streaming import stream_answer
    from doubtdesk.answering.output_guardrails import JudgeVerdict
    from doubtdesk.analytics import tracker

    tracking = Mock()
    monkeypatch.setattr(tracker, "record_openai_usage", tracking)
    provider = Mock()
    provider.responses.stream.return_value = FakeStream("Rejected answer")
    provider.responses.create.return_value = SimpleNamespace(output_text="Repaired answer", usage=SimpleNamespace(input_tokens=8, output_tokens=5))
    judge = Mock(model_name="fake-judge")
    judge.evaluate.side_effect = [
        JudgeVerdict(passed=False, relevance=0.2, groundedness=1, completeness=1, safety="pass", feedback="Try again"),
        JudgeVerdict(passed=True, relevance=1, groundedness=1, completeness=1, safety="pass"),
    ]
    delivered = []

    def content():
        with trace_run("turn"):
            for event, data in stream_answer({"question": "Gravity?", "doc_ids": []}, provider, "fake", "request", "chat", judge=judge):
                if event == "token":
                    yield data["text"]

    async def run():
        response = ContextStreamingResponse(content())
        async def send(message):
            if message["type"] == "http.response.body":
                delivered.append(message["body"])
        await response.stream_response(send)

    anyio.run(run)
    root, _ = assert_single_tree(trace_capture, "turn")
    generations = [span for span in finished(trace_capture) if span.name == "general_answer"]
    checks = [span for span in finished(trace_capture) if span.name == "output_verification"]
    assert len(generations) == len(checks) == 2
    assert all(span.parent.span_id == root.context.span_id for span in generations + checks)
    assert {trace_capture.data[span.context.span_id]["output"] for span in generations} == {"Rejected answer", "Repaired answer"}
    assert b"".join(delivered) == b"Repaired answer"
    assert tracking.call_count == 2


def test_stream_context_when_langfuse_disabled(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    def content():
        with trace_run("turn") as root:
            assert root is None
            with trace_generation("general_answer", "fake") as observation:
                assert observation is None
                observe_llm_response("general_answer", SimpleNamespace(output_text="answer"), "fake")
                yield "answer"
    async def run():
        body = []
        response = ContextStreamingResponse(content())
        async def send(message):
            if message["type"] == "http.response.body":
                body.append(message["body"])
        await response.stream_response(send)
        assert b"".join(body) == b"answer"
    anyio.run(run)



def test_request_inherits_known_upstream_span(trace_capture, caplog):
    upstream_provider = TracerProvider()
    tracer = upstream_provider.get_tracer("upstream-http")

    def content():
        with trace_run("doubtdesk.ask.stream"):
            with trace_generation("general_answer", "fake"):
                yield "answer"
            with trace_stage("save_exchange"):
                pass

    async def run():
        with tracer.start_as_current_span("http.request", context=otel_context.Context()) as upstream:
            expected = upstream.get_span_context()
            response = ContextStreamingResponse(content())
            body = []

            async def send(message):
                if message["type"] == "http.response.body":
                    body.append(message["body"])

            await response.stream_response(send)
            assert b"".join(body) == b"answer"
            assert otel_trace.get_current_span().get_span_context() == expected
            root, spans = assert_single_tree(trace_capture, "doubtdesk.ask.stream", allow_upstream_parent=True)
            assert root.parent == expected
            assert spans["general_answer"].parent.span_id == root.context.span_id
            assert spans["save_exchange"].parent.span_id == root.context.span_id

    try:
        anyio.run(run)
    finally:
        upstream_provider.shutdown()
    assert "Failed to detach context" not in caplog.text

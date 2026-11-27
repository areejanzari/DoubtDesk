import asyncio
import json
import logging

from doubtdesk.answering.schemas import ModelEvidenceResult
from doubtdesk.answering.validation import validate_evidence_consistency, validate_evidence_sources
from doubtdesk.answering.vision import (build_grounded_content, vision_input, agenerate_vision_answer,
    alias_grounded_content, decode_model_evidence, CandidateValidationError, STRUCTURED_ANSWER_MAX_OUTPUT_TOKENS)
from doubtdesk.answering.fallback import general_input, agenerate_general_answer, ANSWER_MAX_OUTPUT_TOKENS
from doubtdesk.answering.formatting import format_answer, normalize_answer_citations
from doubtdesk.answering.json_stream import AnswerFieldStreamer, AliasRewriter
from doubtdesk.answering.output_guardrails import (
    areview_and_repair, is_document_specific, document_insufficient_result,
    VERIFICATION_UNAVAILABLE_MESSAGE,
)
from doubtdesk.db import connection
from doubtdesk.ingestion.registry import get_document_names
from doubtdesk.llm import iterate_with_result, response_stream, run_coroutine
from doubtdesk.observability.tracing import observe_llm_response, trace_generation
from pydantic import ValidationError

logger = logging.getLogger(__name__)

RESULT = "result"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def astream_general(client, model_name: str, question: str, request_id=None, conversation_id=None, *, emit_tokens=True):
    pieces = []
    with trace_generation("general_answer", model_name, request_id, input={"question": question}):
        async with response_stream(
            client,
            model=model_name,
            input=general_input(question),
            reasoning={"effort": "none"},
            max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS,
        ) as stream:
            async for event in stream:
                if event.type == "response.output_text.delta":
                    pieces.append(event.delta)
                    if emit_tokens:
                        yield ("token", {"text": event.delta})
            response = await stream.get_final_response()
            await asyncio.to_thread(observe_llm_response, "general_answer", response, model_name, request_id, conversation_id)
    answer = "".join(pieces)
    if not answer.strip():
        raise RuntimeError("Model returned no answer")
    yield (RESULT, {"status": "general_knowledge", "answer": answer, "sources": []})


async def astream_grounded(client, model_name: str, question: str, pages: list[dict], request_id=None, conversation_id=None, *, chunks=None, allowed_pages=None, content=None, emit_tokens=True):
    if content is None:
        content = await asyncio.to_thread(build_grounded_content, question, chunks or [], pages)
    content, aliases = alias_grounded_content(content)
    pieces = []
    streamer = AnswerFieldStreamer()
    rewriter = AliasRewriter(aliases)
    held = []
    operation = "grounded_visual" if pages else "grounded_text"
    try:
        with trace_generation(operation, model_name, request_id, input={"question": question, "chunk_count": len(chunks or []), "image_count": len(pages)}):
            async with response_stream(
                client,
                model=model_name,
                input=vision_input(content),
                text_format=ModelEvidenceResult,
                reasoning={"effort": "none"},
                max_output_tokens=STRUCTURED_ANSWER_MAX_OUTPUT_TOKENS,
            ) as stream:
                async for event in stream:
                    if event.type != "response.output_text.delta":
                        continue
                    pieces.append(event.delta)
                    if not emit_tokens:
                        continue
                    text = streamer.feed(event.delta)
                    if not text:
                        continue
                    if streamer.status is None:
                        held.append(text)
                        continue
                    if streamer.status != "sufficient":
                        continue
                    visible = rewriter.feed("".join(held) + text)
                    held = []
                    if visible:
                        yield ("token", {"text": visible})
                response = await stream.get_final_response()
                await asyncio.to_thread(observe_llm_response, operation, response, model_name, request_id, conversation_id)
        if emit_tokens and streamer.status == "sufficient":
            tail = rewriter.feed("".join(held)) + rewriter.flush()
            if tail:
                yield ("token", {"text": tail})
        if getattr(response, "status", None) == "incomplete":
            raise CandidateValidationError("Incomplete structured evidence")
        evidence = decode_model_evidence(json.loads("".join(pieces)), aliases)
        validate_evidence_sources(evidence, allowed_pages or pages or chunks or [])
    except (ValueError, TypeError, ValidationError) as exc:
        raise CandidateValidationError(str(exc)) from exc
    except Exception as exc:
        if type(exc).__name__ in {"LengthFinishReasonError", "ContentFilterFinishReasonError"}:
            raise CandidateValidationError("Incomplete structured evidence") from exc
        raise
    if evidence.status == "insufficient":
        yield (RESULT, None)
        return
    yield (RESULT, {
        "status": "grounded",
        "answer": evidence.answer,
        "sources": [source.model_dump() for source in evidence.sources],
    })


def emit_answer_tokens(answer):
    for start in range(0, len(answer), 64):
        yield ("token", {"text": answer[start:start + 64]})


async def review_with_signals(**kwargs):
    signals = asyncio.Queue()
    regenerate = kwargs["regenerate"]

    async def signalling_regenerate(feedback: str) -> dict:
        await signals.put(("revising", feedback))
        return await regenerate(feedback)

    task = asyncio.ensure_future(areview_and_repair(**{**kwargs, "regenerate": signalling_regenerate}))
    try:
        while True:
            getter = asyncio.ensure_future(signals.get())
            done, _ = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield ("signal", getter.result())
                continue
            getter.cancel()
            while not signals.empty():
                yield ("signal", signals.get_nowait())
            yield (RESULT, task.result())
            return
    finally:
        if not task.done():
            task.cancel()


class _Display:
    def __init__(self, judge):
        self.judge = judge
        self.pieces = []
        self.verifying_sent = False

    @property
    def streamed(self):
        return bool(self.pieces)

    def shows(self, answer):
        return self.streamed and "".join(self.pieces) == answer

    def token(self, event):
        events = []
        if self.judge is not None and not self.verifying_sent:
            self.verifying_sent = True
            events.append(("status", {"stage": "verifying"}))
        self.pieces.append(event[1]["text"])
        events.append(event)
        return events

    def reset(self, reason):
        if not self.streamed:
            return []
        self.pieces = []
        return [("reset", {"reason": reason or "replaced"})]

    def replace_with(self, answer, reason):
        if self.shows(answer):
            return []
        events = self.reset(reason)
        events.extend(emit_answer_tokens(answer))
        return events


async def astream_answer(prepared: dict, client, model_name: str, request_id=None, conversation_id=None, *, judge=None):
    if prepared.get("is_off_topic"):
        for event in emit_answer_tokens(prepared["answer"]):
            yield event
        yield (RESULT, {
            "status": "out_of_scope",
            "answer": prepared["answer"],
            "sources": [],
            "formatted_answer": prepared["answer"],
        })
        return

    question = prepared.get("resolved_question", prepared["question"])
    pages = prepared.get("retrieved_pages") or []
    chunks = prepared.get("retrieved_chunks") or []
    low_evidence = prepared.get("evidence_mode") == "low"
    grounded_content = None
    display = _Display(judge)

    if (low_evidence or not chunks) and prepared.get("doc_ids") and is_document_specific(question):
        result = document_insufficient_result()
        for event in emit_answer_tokens(result["answer"]):
            yield event
        yield (RESULT, result)
        return

    result = None
    if chunks and not low_evidence:
        grounded_content = await asyncio.to_thread(build_grounded_content, question, chunks, pages)
        try:
            async for event in astream_grounded(
                client, model_name, question, pages, request_id, conversation_id,
                chunks=chunks, allowed_pages=prepared.get("evidence_pages"), content=grounded_content,
            ):
                if event[0] == RESULT:
                    result = event[1]
                else:
                    for item in display.token(event):
                        yield item
        except CandidateValidationError as exc:
            for item in display.reset("invalid_candidate"):
                yield item
            result = {"status": "grounded", "answer": "", "sources": [], "candidate_error": str(exc)}
        if result is None:
            if prepared.get("doc_ids") and is_document_specific(question):
                result = document_insufficient_result()
                for event in emit_answer_tokens(result["answer"]):
                    yield event
                yield (RESULT, result)
                return
            yield ("status", {"stage": "fallback"})
            async for event in astream_general(client, model_name, question, request_id, conversation_id):
                if event[0] == RESULT:
                    result = event[1]
                else:
                    for item in display.token(event):
                        yield item
    else:
        async for event in astream_general(client, model_name, question, request_id, conversation_id):
            if event[0] == RESULT:
                result = event[1]
            else:
                for item in display.token(event):
                    yield item
    if judge is None and not result.get("candidate_error"):
        if result["status"] == "grounded":
            result["answer"] = normalize_answer_citations(result["answer"], result["sources"])
        for item in display.replace_with(result["answer"], "normalized"):
            yield item
        yield (RESULT, result)
        return

    yield ("status", {"stage": "checking"})
    grounded = result["status"] == "grounded"
    evidence_chunks = chunks if grounded else []
    evidence_pages = pages if grounded else []

    async def regenerate_general(feedback: str) -> dict:
        return {
            "status": "general_knowledge",
            "answer": await agenerate_general_answer(
                client, model_name, question,
                request_id=request_id,
                conversation_id=conversation_id,
                feedback=feedback,
            ),
            "sources": [],
        }

    async def regenerate(feedback: str) -> dict:
        if not grounded:
            return await regenerate_general(feedback)
        content = grounded_content
        if content is None:
            content = await asyncio.to_thread(build_grounded_content, question, chunks, pages)
        evidence = await agenerate_vision_answer(
            client=client,
            model_name=model_name,
            content=content,
            request_id=request_id,
            conversation_id=conversation_id,
            operation="grounded_regeneration",
            feedback=feedback,
        )
        validate_evidence_consistency(evidence)
        validate_evidence_sources(evidence, prepared.get("evidence_pages") or pages or chunks)
        if evidence.status != "sufficient":
            raise CandidateValidationError("Regeneration returned insufficient PDF evidence")
        return {
            "status": "grounded",
            "answer": evidence.answer,
            "sources": [source.model_dump() for source in evidence.sources],
        }

    approved = None
    async for kind, data in review_with_signals(
        result=result, question=question, judge=judge, regenerate=regenerate,
        chunks=evidence_chunks, pages=evidence_pages,
        allowed_pages=prepared.get("evidence_pages") or [],
        request_id=request_id, conversation_id=conversation_id, paraphrases=prepared.get("paraphrases"),
    ):
        if kind == RESULT:
            approved = data
        else:
            for item in display.reset("judge_rejected"):
                yield item
            yield ("status", {"stage": "revising"})
    if approved.get("judge_failure_reason") == "invalid_candidate" and grounded and is_document_specific(question):
        approved = {**approved, **document_insufficient_result()}
    if approved.get("judge_needs_fallback"):
        for item in display.reset(approved.get("judge_failure_reason")):
            yield item
        yield ("status", {"stage": "fallback"})
        first_review = approved
        try:
            fallback_result = None
            async for event in astream_general(client, model_name, question, request_id, conversation_id):
                if event[0] == RESULT:
                    fallback_result = event[1]
                else:
                    for item in display.token(event):
                        yield item
            yield ("status", {"stage": "checking"})
            approved = None
            async for kind, data in review_with_signals(
                result=fallback_result, question=question, judge=judge,
                regenerate=regenerate_general,
                chunks=[], pages=[], allowed_pages=[], request_id=request_id, conversation_id=conversation_id,
                paraphrases=prepared.get("paraphrases"),
            ):
                if kind == RESULT:
                    approved = data
                else:
                    for item in display.reset("judge_rejected"):
                        yield item
                    yield ("status", {"stage": "revising"})
            approved["judge_reviews"] = first_review["judge_reviews"] + approved["judge_reviews"]
            approved["judge_attempts"] += first_review["judge_attempts"]
            approved["judge_regenerations"] += first_review["judge_regenerations"]
            approved["judge_fallback_attempted"] = True
        except Exception:
            logger.exception("General-knowledge fallback generation failed")
            approved = {
                **first_review,
                "status": "verification_failed",
                "answer": VERIFICATION_UNAVAILABLE_MESSAGE,
                "sources": [],
                "judge_failure_reason": "fallback_generation_error",
                "judge_needs_fallback": False,
                "judge_fallback_attempted": True,
                "verification_failed": True,
            }
    if approved["status"] == "grounded":
        approved["answer"] = normalize_answer_citations(approved["answer"], approved["sources"])
    for item in display.replace_with(approved["answer"], approved.get("judge_failure_reason")):
        yield item
    yield (RESULT, approved)


def stream_general(*args, **kwargs):
    return (yield from iterate_with_result(astream_general(*args, **kwargs), RESULT))


async def _grounded_result(agen):
    result = None
    async for event in agen:
        if event[0] == RESULT:
            result = event[1]
    return result


def stream_grounded(*args, **kwargs):
    return run_coroutine(_grounded_result(astream_grounded(*args, **kwargs)))


def stream_answer(*args, **kwargs):
    return (yield from iterate_with_result(astream_answer(*args, **kwargs), RESULT))


def finalize_stream_result(result: dict, *, user_id: str | None = None, normalized=False):
    if result["status"] == "out_of_scope":
        return result
    with connection() as conn:
        names = get_document_names(conn, [source["doc_id"] for source in result["sources"]], user_id=user_id) if result["sources"] else {}
    if result["status"] == "grounded" and not normalized:
        result["answer"] = normalize_answer_citations(result["answer"], result["sources"])
    result["formatted_answer"] = format_answer(result, names, normalized=True)
    return result


async def ainvoke_graph(graph, state, config=None):
    ainvoke = getattr(type(graph), "ainvoke", None)
    if ainvoke is not None and asyncio.iscoroutinefunction(ainvoke):
        return await graph.ainvoke(state, config=config)
    return await asyncio.to_thread(graph.invoke, state, config=config)


class PreparedAnswerGraph:
    """Compatibility invoke interface for /ask and full evaluation; same answering path as SSE."""
    def __init__(self, preparation_graph, client, model_name, judge):
        self.preparation_graph = preparation_graph
        self.client = client
        self.model_name = model_name
        self.judge = judge

    def invoke(self, state, config=None):
        prepared = self.preparation_graph.invoke(state, config=config)
        generator = stream_answer(prepared, self.client, self.model_name,
                                  state.get("request_id"), state.get("conversation_id"), judge=self.judge)
        try:
            while True:
                next(generator)
        except StopIteration as finished:
            result = finalize_stream_result(finished.value,
                                            user_id=state["user_id"], normalized=True)
            return {**prepared, **result}
        finally:
            generator.close()

    async def ainvoke(self, state, config=None):
        prepared = await ainvoke_graph(self.preparation_graph, state, config)
        result = None
        async for event, data in astream_answer(prepared, self.client, self.model_name,
                                                state.get("request_id"), state.get("conversation_id"), judge=self.judge):
            if event == RESULT:
                result = data
        result = await asyncio.to_thread(finalize_stream_result, result, user_id=state["user_id"], normalized=True)
        return {**prepared, **result}

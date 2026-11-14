from doubtdesk.config import settings
import json
import logging
import sqlite3
from contextlib import closing

from doubtdesk.answering.schemas import ModelEvidenceResult
from doubtdesk.answering.validation import validate_evidence_consistency, validate_evidence_sources
from doubtdesk.answering.vision import (build_grounded_content, vision_input, generate_vision_answer,
    alias_grounded_content, decode_model_evidence, CandidateValidationError, STRUCTURED_ANSWER_MAX_OUTPUT_TOKENS)
from doubtdesk.answering.fallback import general_input, generate_general_answer, ANSWER_MAX_OUTPUT_TOKENS
from doubtdesk.answering.formatting import format_answer, normalize_answer_citations
from doubtdesk.answering.output_guardrails import (
    review_and_repair, is_document_specific, document_insufficient_result,
    VERIFICATION_UNAVAILABLE_MESSAGE,
)
from doubtdesk.ingestion.registry import get_document_names
from doubtdesk.observability.tracing import observe_llm_response, trace_generation
from pydantic import ValidationError

logger = logging.getLogger(__name__)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_general(client, model_name: str, question: str, request_id=None, conversation_id=None, *, emit_tokens=True):
    pieces = []
    with trace_generation("general_answer", model_name, request_id, input={"question": question}):
        with client.responses.stream(
            model=model_name,
            input=general_input(question),
            reasoning={"effort": "none"},
            max_output_tokens=ANSWER_MAX_OUTPUT_TOKENS,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    pieces.append(event.delta)
                    if emit_tokens:
                        yield ("token", {"text": event.delta})
            response = stream.get_final_response()
            observe_llm_response("general_answer", response, model_name, request_id, conversation_id)
    answer = "".join(pieces)
    if not answer.strip():
        raise RuntimeError("Model returned no answer")
    return {"status": "general_knowledge", "answer": answer, "sources": []}


def stream_grounded(client, model_name: str, question: str, pages: list[dict], request_id=None, conversation_id=None, *, chunks=None, allowed_pages=None):
    content, aliases = alias_grounded_content(build_grounded_content(question, chunks or [], pages))
    pieces = []
    operation = "grounded_visual" if pages else "grounded_text"
    try:
        with trace_generation(operation, model_name, request_id, input={"question": question, "chunk_count": len(chunks or []), "image_count": len(pages)}):
            with client.responses.stream(
                model=model_name,
                input=vision_input(content),
                text_format=ModelEvidenceResult,
                reasoning={"effort": "none"},
                max_output_tokens=STRUCTURED_ANSWER_MAX_OUTPUT_TOKENS,
            ) as stream:
                for event in stream:
                    if event.type == "response.output_text.delta":
                        pieces.append(event.delta)
                response = stream.get_final_response()
                observe_llm_response(operation, response, model_name, request_id, conversation_id)
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
        return None
    return {
        "status": "grounded",
        "answer": evidence.answer,
        "sources": [source.model_dump() for source in evidence.sources],
    }


def emit_answer_tokens(answer):
    for start in range(0, len(answer), 64):
        yield ("token", {"text": answer[start:start + 64]})


def stream_answer(prepared: dict, client, model_name: str, request_id=None, conversation_id=None, *, judge=None):
    if prepared.get("is_off_topic"):
        yield from emit_answer_tokens(prepared["answer"])
        return {
            "status": "out_of_scope",
            "answer": prepared["answer"],
            "sources": [],
            "formatted_answer": prepared["answer"],
        }

    question = prepared.get("resolved_question", prepared["question"])
    pages = prepared.get("retrieved_pages") or []
    chunks = prepared.get("retrieved_chunks") or []
    low_evidence = prepared.get("evidence_mode") == "low"

    if (low_evidence or not chunks) and prepared.get("doc_ids") and is_document_specific(question):
        result = document_insufficient_result()
        yield from emit_answer_tokens(result["answer"])
        return result

    if chunks and not low_evidence:
        try:
            result = stream_grounded(
                client, model_name, question, pages, request_id, conversation_id,
                chunks=chunks, allowed_pages=prepared.get("evidence_pages"),
            )
        except CandidateValidationError as exc:
            result = {"status": "grounded", "answer": "", "sources": [], "candidate_error": str(exc)}
        if result is None:
            if prepared.get("doc_ids") and is_document_specific(question):
                result = document_insufficient_result()
                yield from emit_answer_tokens(result["answer"])
                return result
            yield ("status", {"stage": "fallback"})
            result = yield from stream_general(
                client, model_name, question, request_id, conversation_id,
                emit_tokens=judge is None,
            )
    else:
        result = yield from stream_general(
            client, model_name, question, request_id, conversation_id,
            emit_tokens=judge is None,
        )

    if judge is None and not result.get("candidate_error"):
        if result["status"] == "grounded":
            result["answer"] = normalize_answer_citations(result["answer"], result["sources"])
            yield from emit_answer_tokens(result["answer"])
        return result

    yield ("status", {"stage": "checking"})
    grounded = result["status"] == "grounded"
    evidence_chunks = chunks if grounded else []
    evidence_pages = pages if grounded else []

    def regenerate(feedback: str) -> dict:
        if grounded:
            content = build_grounded_content(question, chunks, pages)
            evidence = generate_vision_answer(
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
        return {
            "status": "general_knowledge",
            "answer": generate_general_answer(
                client, model_name, question,
                request_id=request_id,
                conversation_id=conversation_id,
                feedback=feedback,
            ),
            "sources": [],
        }

    approved = review_and_repair(
        result=result, question=question, judge=judge, regenerate=regenerate,
        chunks=evidence_chunks, pages=evidence_pages,
        allowed_pages=prepared.get("evidence_pages") or [],
        request_id=request_id, conversation_id=conversation_id,
    )
    if approved.get("judge_failure_reason") == "invalid_candidate" and grounded and is_document_specific(question):
        approved = {**approved, **document_insufficient_result()}
    if approved.get("judge_needs_fallback"):
        yield ("status", {"stage": "fallback"})
        first_review = approved
        try:
            fallback_result = yield from stream_general(
                client, model_name, question, request_id, conversation_id,
                emit_tokens=False,
            )
            approved = review_and_repair(
                result=fallback_result, question=question, judge=judge,
                regenerate=lambda feedback: {
                    "status": "general_knowledge",
                    "answer": generate_general_answer(
                        client, model_name, question,
                        request_id=request_id, conversation_id=conversation_id,
                        feedback=feedback,
                    ),
                    "sources": [],
                },
                chunks=[], pages=[], allowed_pages=[], request_id=request_id, conversation_id=conversation_id,
            )
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
    yield from emit_answer_tokens(approved["answer"])
    return approved


def finalize_stream_result(result: dict, db_path=settings.documents_db, *, user_id: str | None = None, normalized=False):
    if result["status"] == "out_of_scope":
        return result
    with closing(sqlite3.connect(db_path)) as conn:
        names = get_document_names(conn, [source["doc_id"] for source in result["sources"]], user_id=user_id) if result["sources"] else {}
    if result["status"] == "grounded" and not normalized:
        result["answer"] = normalize_answer_citations(result["answer"], result["sources"])
    result["formatted_answer"] = format_answer(result, names, normalized=True)
    return result


class PreparedAnswerGraph:
    """Compatibility invoke interface for /ask and full evaluation; same answering path as SSE."""
    def __init__(self, preparation_graph, client, model_name, judge, db_path=settings.documents_db):
        self.preparation_graph = preparation_graph
        self.client = client
        self.model_name = model_name
        self.judge = judge
        self.db_path = db_path

    def invoke(self, state, config=None):
        prepared = self.preparation_graph.invoke(state, config=config)
        generator = stream_answer(prepared, self.client, self.model_name,
                                  state.get("request_id"), state.get("conversation_id"), judge=self.judge)
        try:
            while True:
                next(generator)
        except StopIteration as finished:
            result = finalize_stream_result(finished.value, db_path=self.db_path,
                                            user_id=state["user_id"], normalized=True)
            return {**prepared, **result}
        finally:
            generator.close()

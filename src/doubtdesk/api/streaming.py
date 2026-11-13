import json
import logging
import sqlite3
from contextlib import closing

from doubtdesk.answering.schemas import EvidenceResult
from doubtdesk.answering.validation import validate_evidence_consistency, validate_evidence_sources
from doubtdesk.answering.vision import build_grounded_content, vision_input, generate_vision_answer
from doubtdesk.answering.fallback import general_input, generate_general_answer
from doubtdesk.answering.formatting import format_answer
from doubtdesk.answering.output_guardrails import (
    review_and_repair, is_document_specific, document_insufficient_result,
    VERIFICATION_UNAVAILABLE_MESSAGE,
)
from doubtdesk.ingestion.registry import get_document_names
from doubtdesk.api.stream_parser import IncrementalEvidenceAnswer
from doubtdesk.observability.tracing import observe_llm_response

logger = logging.getLogger(__name__)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def stream_general(client, model_name: str, question: str, request_id=None, conversation_id=None, *, emit_tokens=True):
    pieces = []
    with client.responses.stream(
        model=model_name,
        input=general_input(question),
        reasoning={"effort": "none"},
        max_output_tokens=300,
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


def stream_grounded(client, model_name: str, question: str, pages: list[dict], request_id=None, conversation_id=None, *, chunks=None, allowed_pages=None, emit_tokens=True):
    content = build_grounded_content(question, chunks or [], pages)
    parser = IncrementalEvidenceAnswer()
    with client.responses.stream(
        model=model_name,
        input=vision_input(content),
        text_format=EvidenceResult,
        reasoning={"effort": "none"},
        max_output_tokens=300,
    ) as stream:
        for event in stream:
            if event.type == "response.output_text.delta":
                delta = parser.feed(event.delta)
                if emit_tokens and delta:
                    yield ("token", {"text": delta})
        response = stream.get_final_response()
        observe_llm_response("grounded_visual" if pages else "grounded_text", response, model_name, request_id, conversation_id)
    evidence = EvidenceResult.model_validate_json(parser.raw)
    validate_evidence_consistency(evidence)
    validate_evidence_sources(evidence, allowed_pages or pages)
    if evidence.status == "insufficient":
        return None
    return {
        "status": "grounded",
        "answer": evidence.answer,
        "sources": [source.model_dump() for source in evidence.sources],
    }


def stream_answer(prepared: dict, client, model_name: str, request_id=None, conversation_id=None, *, judge=None):
    if prepared.get("is_off_topic"):
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
        yield ("token", {"text": result["answer"]})
        return result

    if chunks and not low_evidence:
        result = yield from stream_grounded(
            client, model_name, question, pages, request_id, conversation_id,
            chunks=chunks, allowed_pages=prepared.get("evidence_pages"),
            emit_tokens=judge is None,
        )
        if result is None:
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

    if judge is None:
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
            validate_evidence_sources(evidence, prepared.get("evidence_pages") or pages)
            if evidence.status != "sufficient":
                raise ValueError("Regeneration returned insufficient PDF evidence")
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
    )
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
                chunks=[], pages=[], allowed_pages=[],
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
    for start in range(0, len(approved["answer"]), 64):
        yield ("token", {"text": approved["answer"][start:start + 64]})
    return approved


def finalize_stream_result(result: dict, db_path="data/documents.db"):
    if result["status"] == "out_of_scope":
        return result
    with closing(sqlite3.connect(db_path)) as conn:
        names = get_document_names(conn, [source["doc_id"] for source in result["sources"]])
    result["formatted_answer"] = format_answer(result, names)
    return result

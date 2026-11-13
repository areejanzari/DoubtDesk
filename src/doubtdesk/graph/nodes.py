import os
import sqlite3
from contextlib import closing
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.ingestion.registry import get_document_names, validate_user_document_ids
from doubtdesk.retrieval.hybrid import hybrid_search, forward_evidence_chunks
from doubtdesk.retrieval.pages import get_page_image_paths
from doubtdesk.answering.vision import build_grounded_content, generate_vision_answer
from doubtdesk.answering.evidence_selection import (
    assess_text_evidence,
    choose_evidence_route,
)
from doubtdesk.answering.validation import validate_evidence_sources, validate_evidence_consistency
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.formatting import format_answer
from doubtdesk.guardrails.paraphrasing import resolve_and_paraphrase
from doubtdesk.memory.vector import search_memory
from doubtdesk.guardrails.responses import get_off_topic_response
from doubtdesk.answering.output_guardrails import (
    review_and_repair, is_document_specific, document_insufficient_result,
    VERIFICATION_UNAVAILABLE_MESSAGE,
)


def retrieve_node(collection, embedding_model, db_path):
    def retrieve(state: DoubtDeskState) -> dict:
        question = state["question"]
        doc_ids = state["doc_ids"]
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not doc_ids:
            return {"retrieved_chunks": [], "retrieved_pages": []}
        with closing(sqlite3.connect(db_path)) as conn:
            validate_user_document_ids(conn, user_id=state["user_id"], doc_ids=doc_ids)
        ranked_chunks = hybrid_search(
            collection,
            embedding_model,
            state.get("paraphrases") or [question],
            doc_ids,
        )
        return {"retrieved_chunks": forward_evidence_chunks(ranked_chunks), "retrieved_pages": [], "retrieval_candidate_count": len(ranked_chunks)}
    return retrieve


def evidence_selection_node(client, model_name):
    def select(state: DoubtDeskState) -> dict:
        chunks = state["retrieved_chunks"]
        paraphrases = state["paraphrases"]
        assessment = assess_text_evidence(
            client,
            model_name,
            state.get("resolved_question", state["question"]),
            paraphrases,
            chunks,
            state.get("request_id"),
            state.get("conversation_id"),
        )
        selection = choose_evidence_route(assessment, paraphrases, chunks)
        if selection["evidence_mode"] == "visual":
            selection["retrieved_pages"] = get_page_image_paths(selection["retrieved_pages"])
        selection["evidence_assessment"] = assessment
        return selection
    return select


def evidence_node(client, model_name):
    def evaluate_evidence(state: DoubtDeskState) -> dict:
        chunks = state.get("retrieved_chunks") or []
        pages = state.get("retrieved_pages") or []
        allowed_pages = state.get("evidence_pages") or []
        if not chunks and not pages:
            raise ValueError("Evidence evaluation requires retrieved text or pages")
        content = build_grounded_content(
            question=state.get("resolved_question", state["question"]),
            chunks=chunks,
            pages=pages,
        )
        evidence = generate_vision_answer(
            client=client,
            model_name=model_name,
            content=content,
            request_id=state.get("request_id"),
            conversation_id=state.get("conversation_id"),
            operation=f"grounded_{state.get('evidence_mode', 'visual')}",
        )
        validate_evidence_consistency(evidence)
        validate_evidence_sources(evidence, allowed_pages or pages)
        return {"evidence": evidence}
    return evaluate_evidence


def fallback_node(client, model_name):
    def generate_fallback(state: DoubtDeskState) -> dict:
        question = state.get("resolved_question", state["question"])
        from_rejected_grounding = bool(state.get("judge_needs_fallback"))
        if state.get("doc_ids") and is_document_specific(question):
            return document_insufficient_result()
        try:
            answer = generate_general_answer(
                client=client,
                model_name=model_name,
                question=question,
                request_id=state.get("request_id"),
                conversation_id=state.get("conversation_id"),
            )
        except Exception:
            if not from_rejected_grounding:
                raise
            return {
                "status": "verification_failed", "answer": VERIFICATION_UNAVAILABLE_MESSAGE,
                "sources": [], "judge_needs_fallback": False,
                "judge_fallback_attempted": True, "skip_output_judge": True,
                "judge_failure_reason": "fallback_generation_error",
                "verification_failed": True,
            }
        response = {"status": "general_knowledge", "answer": answer, "sources": []}
        if from_rejected_grounding:
            response.update({
                "judge_needs_fallback": False,
                "judge_fallback_attempted": True,
                "skip_output_judge": False,
                "verification_failed": False,
            })
        return response
    return generate_fallback


def insufficient_document_node(state: DoubtDeskState) -> dict:
    return document_insufficient_result()


def route_selected_evidence(state: DoubtDeskState) -> str:
    if state.get("evidence_mode") != "low":
        return "evidence"
    question = state.get("resolved_question", state["question"])
    return "insufficient_document" if is_document_specific(question) else "fallback"


def output_guardrail_node(client, model_name, judge):
    def review(state: DoubtDeskState) -> dict:
        if state.get("skip_output_judge") or state.get("status") == "out_of_scope":
            return {}
        question = state.get("resolved_question", state["question"])
        chunks = state.get("retrieved_chunks") or []
        pages = state.get("retrieved_pages") or []
        allowed_pages = state.get("evidence_pages") or []
        result = {
            "status": state["status"],
            "answer": state["answer"],
            "sources": state["sources"],
        }

        def regenerate(feedback: str) -> dict:
            if result["status"] == "grounded":
                content = build_grounded_content(question, chunks, pages)
                evidence = generate_vision_answer(
                    client=client,
                    model_name=model_name,
                    content=content,
                    request_id=state.get("request_id"),
                    conversation_id=state.get("conversation_id"),
                    operation="grounded_regeneration",
                    feedback=feedback,
                )
                validate_evidence_consistency(evidence)
                validate_evidence_sources(evidence, allowed_pages)
                if evidence.status != "sufficient":
                    raise ValueError("Regeneration did not find sufficient PDF evidence")
                return {
                    "status": "grounded",
                    "answer": evidence.answer,
                    "sources": [source.model_dump() for source in evidence.sources],
                }
            return {
                "status": "general_knowledge",
                "answer": generate_general_answer(
                    client, model_name, question,
                    request_id=state.get("request_id"),
                    conversation_id=state.get("conversation_id"),
                    feedback=feedback,
                ),
                "sources": [],
            }

        reviewed = review_and_repair(
            result=result, question=question, judge=judge, regenerate=regenerate,
            chunks=chunks if result["status"] == "grounded" else [],
            pages=pages if result["status"] == "grounded" else [],
            allowed_pages=allowed_pages,
        )
        if state.get("judge_fallback_attempted"):
            reviewed["judge_reviews"] = (state.get("judge_reviews") or []) + reviewed["judge_reviews"]
            reviewed["judge_attempts"] += state.get("judge_attempts", 0)
            reviewed["judge_regenerations"] += state.get("judge_regenerations", 0)
        return reviewed
    return review


def route_after_output_guardrail(state: DoubtDeskState) -> str:
    if state.get("judge_needs_fallback") and not state.get("judge_fallback_attempted"):
        return "fallback"
    return "format"


def grounded_node(state: DoubtDeskState) -> dict:
    evidence = state["evidence"]
    if evidence.status != "sufficient":
        raise ValueError("Grounded answer requires sufficient evidence")
    return {
        "status": "grounded",
        "answer": evidence.answer,
        "sources": [source.model_dump() for source in evidence.sources],
    }


def formatting_node(db_path):
    def format_response(state: DoubtDeskState) -> dict:
        response = {"status": state["status"], "answer": state["answer"], "sources": state["sources"]}
        cited_doc_ids = [source["doc_id"] for source in response["sources"]]
        with sqlite3.connect(db_path) as conn:
            document_names = get_document_names(conn, cited_doc_ids)
        return {"formatted_answer": format_answer(response, document_names)}
    return format_response


def paraphrase_node(client, model_name, embedding_model=None, memory_collection=None):
    def paraphrase(state: DoubtDeskState) -> dict:
        question = state["question"]
        conversation_id = state.get("conversation_id")
        user_id = state["user_id"]
        history = []
        if conversation_id:
            history = search_memory(
                conversation_id=conversation_id,
                user_id=user_id,
                question=question,
                doc_ids=state.get("doc_ids", []),
                model=embedding_model,
                collection=memory_collection,
            )
        result = resolve_and_paraphrase(
            client=client,
            model_name=model_name,
            question=question,
            history=history,
            request_id=state.get("request_id"),
            conversation_id=conversation_id,
        )
        return {"resolved_question": result.resolved_question, "paraphrases": result.questions}
    return paraphrase


def scope_node(classifier):
    def classify(state: DoubtDeskState) -> dict:
        predictions = classifier.predict(state["paraphrases"])
        return {
            "scope_predictions": predictions,
            "is_off_topic": all(prediction["label"] == 1 for prediction in predictions),
        }
    return classify


def rejection_node(state: DoubtDeskState) -> dict:
    message = get_off_topic_response()
    return {
        "status": "out_of_scope",
        "answer": message,
        "sources": [],
        "formatted_answer": message,
    }

import asyncio
from doubtdesk.config import runtime_settings
from doubtdesk.db import connection
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.ingestion.registry import validate_user_document_ids
from doubtdesk.ingestion.figures import question_refs
from doubtdesk.llm import run_coroutine
from doubtdesk.retrieval.hybrid import aretrieve_ranked_evidence, retrieval_queries
from doubtdesk.retrieval.pages import get_page_image_paths
from doubtdesk.retrieval.vector_store import page_metadata, pages_with_figure_refs
from doubtdesk.answering.evidence_selection import (
    aassess_text_evidence,
    choose_evidence_route,
    choose_evidence_route_rules,
)
from doubtdesk.guardrails.paraphrasing import aresolve_and_paraphrase
from doubtdesk.memory.context import abuild_history
from doubtdesk.guardrails.responses import get_off_topic_response


def _sync(async_node):
    def node(state: DoubtDeskState) -> dict:
        return run_coroutine(async_node(state))
    return node


def aretrieve_node(embedding_model, reranker=None):
    async def retrieve(state: DoubtDeskState) -> dict:
        question = state["question"]
        doc_ids = state["doc_ids"]
        if not question.strip():
            raise ValueError("Question cannot be empty")
        if not doc_ids:
            return {"retrieved_chunks": [], "retrieved_pages": []}

        def validate():
            with connection() as conn:
                validate_user_document_ids(conn, user_id=state["user_id"], doc_ids=doc_ids)

        await asyncio.to_thread(validate)
        resolved = state.get("resolved_question") or question
        queries = retrieval_queries(resolved, state.get("paraphrases") or [question])
        ranked, forwarded = await aretrieve_ranked_evidence(
            embedding_model, queries, doc_ids, question=resolved, reranker=reranker,
        )
        scores = [chunk["rerank_score"] for chunk in forwarded if "rerank_score" in chunk and not chunk.get("context_only")]
        return {
            "retrieved_chunks": forwarded,
            "retrieved_pages": [],
            "retrieval_candidate_count": len(ranked),
            "retrieval_top_score": max(scores) if scores else None,
        }
    return retrieve


def retrieve_node(embedding_model, reranker=None):
    return _sync(aretrieve_node(embedding_model, reranker))


async def _rule_selection(state: DoubtDeskState) -> dict:
    chunks = state.get("retrieved_chunks") or []
    question = state.get("resolved_question", state["question"])
    refs = question_refs(question)
    referenced = []
    if refs and state.get("doc_ids"):
        try:
            referenced = await asyncio.to_thread(pages_with_figure_refs, state["doc_ids"], refs)
        except Exception:
            referenced = []
    keys = list(dict.fromkeys([(chunk["doc_id"], int(chunk["page_number"])) for chunk in chunks] + referenced))
    try:
        meta = await asyncio.to_thread(page_metadata, keys)
    except Exception:
        meta = {}
    return choose_evidence_route_rules(question, chunks, meta, state.get("retrieval_top_score"), referenced)


def aevidence_selection_node(client, model_name):
    async def select(state: DoubtDeskState) -> dict:
        chunks = state["retrieved_chunks"]
        paraphrases = state["paraphrases"]
        if runtime_settings().evidence_assessment_enabled:
            assessment = await aassess_text_evidence(
                client,
                model_name,
                state.get("resolved_question", state["question"]),
                paraphrases,
                chunks,
                state.get("request_id"),
                state.get("conversation_id"),
            )
            selection = choose_evidence_route(assessment, paraphrases, chunks)
            selection["evidence_assessment"] = assessment
        else:
            selection = await _rule_selection(state)
        if selection["evidence_mode"] == "visual":
            selection["retrieved_pages"] = await asyncio.to_thread(get_page_image_paths, selection["retrieved_pages"])
        return selection
    return select


def evidence_selection_node(client, model_name):
    return _sync(aevidence_selection_node(client, model_name))


def aparaphrase_node(client, model_name, embedding_model=None):
    async def paraphrase(state: DoubtDeskState) -> dict:
        question = state["question"]
        conversation_id = state.get("conversation_id")
        history = []
        if conversation_id:
            history = await abuild_history(
                conversation_id=conversation_id,
                user_id=state["user_id"],
                question=question,
                doc_ids=state.get("doc_ids", []),
                model=embedding_model,
            )
        result = await aresolve_and_paraphrase(
            client=client,
            model_name=model_name,
            question=question,
            history=history,
            request_id=state.get("request_id"),
            conversation_id=conversation_id,
        )
        return {"resolved_question": result.resolved_question, "paraphrases": result.questions}
    return paraphrase


def paraphrase_node(client, model_name, embedding_model=None):
    return _sync(aparaphrase_node(client, model_name, embedding_model))


def scope_node(classifier):
    def classify(state: DoubtDeskState) -> dict:
        predictions = classifier.predict(state["paraphrases"])
        return {
            "scope_predictions": predictions,
            "is_off_topic": all(prediction["label"] == 1 for prediction in predictions),
        }
    return classify


def ascope_node(classifier):
    sync_classify = scope_node(classifier)

    async def classify(state: DoubtDeskState) -> dict:
        return await asyncio.to_thread(sync_classify, state)
    return classify


def rejection_node(state: DoubtDeskState) -> dict:
    message = get_off_topic_response()
    return {
        "status": "out_of_scope",
        "answer": message,
        "sources": [],
        "formatted_answer": message,
    }

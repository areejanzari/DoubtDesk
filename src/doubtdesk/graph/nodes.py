import sqlite3
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.ingestion.registry import (
    get_document_page_counts,
    validate_document_ids,
)
from doubtdesk.retrieval.search import (
    embed_queries,
    search_chunks,
    format_search_results,
)
from doubtdesk.retrieval.pages import (
    group_chunks_by_page,
    expand_neighbor_pages,
    get_page_image_paths,
)

from doubtdesk.answering.vision import (
    build_vision_content,
    generate_vision_answer,
)
from doubtdesk.answering.validation import (
    validate_evidence_sources,
    validate_evidence_consistency,
)
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.formatting import format_answer
from doubtdesk.ingestion.registry import get_document_names
from doubtdesk.guardrails.paraphrasing import resolve_and_paraphrase
from doubtdesk.memory.context import get_conversation_context
from doubtdesk.guardrails.responses import get_off_topic_response


def retrieve_node(collection, embedding_model, db_path):
    def retrieve(state: DoubtDeskState) -> dict:
        question = state["question"]
        doc_ids = state["doc_ids"]

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not doc_ids:
            return {"retrieved_pages": []}

        paraphrases = state.get("paraphrases") or [question]

        with sqlite3.connect(db_path) as conn:
            validate_document_ids(conn, doc_ids)

            question_embeddings = embed_queries(
                paraphrases,
                embedding_model,
            )

            results = search_chunks(
                collection,
                question_embeddings,
                doc_ids,
            )

            chunks = format_search_results(results)

            unique_chunks = {}

            for chunk in chunks:
                chunk_id = chunk["chunk_id"]

                if (
                    chunk_id not in unique_chunks
                    or chunk["distance"] < unique_chunks[chunk_id]["distance"]
                ):
                    unique_chunks[chunk_id] = chunk

            ranked_chunks = sorted(
                unique_chunks.values(),
                key=lambda chunk: chunk["distance"],
            )

            matched_pages = group_chunks_by_page(
                ranked_chunks
            )

            page_counts = get_document_page_counts(
                conn,
                doc_ids,
            )

            selected_pages = expand_neighbor_pages(
                matched_pages[:5],
                page_counts,
                radius=1,
                max_pages=6,
            )

            pages = get_page_image_paths(selected_pages)

        return {"retrieved_pages": pages}

    return retrieve


def evidence_node(client, model_name):
    def evaluate_evidence(state: DoubtDeskState) -> dict:
        pages = state["retrieved_pages"]

        if not pages:
            raise ValueError(
                "Evidence evaluation requires retrieved pages"
            )

        content = build_vision_content(
            question=state.get("resolved_question", state["question"]),
            pages=pages,
        )

        evidence = generate_vision_answer(
            client=client,
            model_name=model_name,
            content=content,
            request_id=state.get("request_id"),
            conversation_id=state.get("conversation_id"),
        )

        validate_evidence_consistency(evidence)
        validate_evidence_sources(evidence, pages)

        return {"evidence": evidence}

    return evaluate_evidence



def fallback_node(client, model_name):
    def generate_fallback(state: DoubtDeskState) -> dict:
        answer = generate_general_answer(
            client=client,
            model_name=model_name,
            question=state.get("resolved_question", state["question"]),
            request_id=state.get("request_id"),
            conversation_id=state.get("conversation_id"),
        )

        return {
            "status": "general_knowledge",
            "answer": answer,
            "sources": [],
        }

    return generate_fallback


def grounded_node(state: DoubtDeskState) -> dict:
    evidence = state["evidence"]

    if evidence.status != "sufficient":
        raise ValueError(
            "Grounded answer requires sufficient evidence"
        )

    return {
        "status": "grounded",
        "answer": evidence.answer,
        "sources": [
            source.model_dump()
            for source in evidence.sources
        ],
    }


def formatting_node(db_path):
    def format_response(state: DoubtDeskState) -> dict:
        response = {
            "status": state["status"],
            "answer": state["answer"],
            "sources": state["sources"],
        }

        cited_doc_ids = [
            source["doc_id"]
            for source in response["sources"]
        ]

        with sqlite3.connect(db_path) as conn:
            document_names = get_document_names(
                conn,
                cited_doc_ids,
            )

        formatted_answer = format_answer(
            response,
            document_names,
        )

        return {"formatted_answer": formatted_answer}

    return format_response


def paraphrase_node(client, model_name):
    def paraphrase(state: DoubtDeskState) -> dict:
        question = state["question"]
        conversation_id = state.get("conversation_id")

        history = []

        if conversation_id:
            history = get_conversation_context(
                conversation_id=conversation_id,
                question=question,
                doc_ids=state.get("doc_ids", []),
            )

        result = resolve_and_paraphrase(
            client=client,
            model_name=model_name,
            question=question,
            history=history,
            request_id=state.get("request_id"),
            conversation_id=state.get("conversation_id"),
        )

        return {
            "resolved_question": result.resolved_question,
            "paraphrases": result.questions,
        }

    return paraphrase


def scope_node(classifier):
    def classify(state: DoubtDeskState) -> dict:
        predictions = classifier.predict(
            state["paraphrases"]
        )

        is_off_topic = all(
            prediction["label"] == 1
            for prediction in predictions
        )

        return {
            "scope_predictions": predictions,
            "is_off_topic": is_off_topic,
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
import sqlite3
from doubtdesk.graph.state import DoubtDeskState
from doubtdesk.ingestion.registry import (
    get_document_page_counts,
    validate_document_ids,
)
from doubtdesk.retrieval.search import (
    embed_query,
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




def retrieve_node(collection, embedding_model, db_path):
    def retrieve(state: DoubtDeskState) -> dict:
        question = state["question"]
        doc_ids = state["doc_ids"]

        if not question.strip():
            raise ValueError("Question cannot be empty")

        if not doc_ids:
            return {"retrieved_pages": []}

        with sqlite3.connect(db_path) as conn:
            validate_document_ids(conn, doc_ids)

            question_embedding = embed_query(
                question, embedding_model
            )

            results = search_chunks(
                collection,
                question_embedding,
                doc_ids,
            )

            chunks = format_search_results(results)
            matched_pages = group_chunks_by_page(chunks)

            page_counts = get_document_page_counts(
                conn, doc_ids
            )

            selected_pages = expand_neighbor_pages(
                matched_pages,
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
            question=state["question"],
            pages=pages,
        )

        evidence = generate_vision_answer(
            client=client,
            model_name=model_name,
            content=content,
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
            question=state["question"],
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

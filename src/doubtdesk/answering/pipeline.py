import sqlite3
from chromadb.api.models.Collection import Collection
from google import genai
from sentence_transformers import SentenceTransformer
from doubtdesk.answering.formatting import format_answer
from doubtdesk.answering.service import answer_question
from doubtdesk.ingestion.registry import (
    get_document_names,
    get_document_page_counts,
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


def run_answering_pipeline(
    question: str,
    doc_ids: list[str],
    collection: Collection,
    embedding_model: SentenceTransformer,
    conn: sqlite3.Connection,
    client: genai.Client,
    model_name: str,
) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    if doc_ids:
        question_embedding = embed_query(
            question, embedding_model
        )

        search_results = search_chunks(
            collection,
            question_embedding,
            doc_ids,
        )

        chunks = format_search_results(search_results)
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
    else:
        pages = []

    response = answer_question(
        question=question,
        pages=pages,
        client=client,
        model_name=model_name,
    )

    cited_doc_ids = [
        source["doc_id"]
        for source in response["sources"]
    ]

    document_names = get_document_names(
        conn, cited_doc_ids
    )

    formatted_answer = format_answer(
        response, document_names
    )

    return {
        **response,
        "formatted_answer": formatted_answer,
    }

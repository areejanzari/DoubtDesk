import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

def embed_queries(
    questions: list[str],
    model: SentenceTransformer,
) -> np.ndarray:
    if not questions or any(not question.strip() for question in questions):
        raise ValueError("Questions cannot be empty")

    return model.encode(
        questions,
        normalize_embeddings=True,
    )

def search_chunks(
    collection: Collection,
    question_embedding: np.ndarray,
    doc_ids: list[str],
    top_k: int = 5,
) -> dict:
    if not doc_ids:
        raise ValueError("Atleast one document ID is required.")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    return collection.query(
        query_embeddings=np.atleast_2d(question_embedding).tolist(),
        n_results=top_k,
        where={"doc_id": {"$in": doc_ids}},
        include=["documents", "metadatas", "distances"],
    )


def format_search_results(results: dict) -> list[dict]:
    formatted = []

    for ids, texts, metadatas, distances in zip(
        results["ids"],
        results["documents"],
        results["metadatas"],
        results["distances"],
        strict=True,
    ):
        for chunk_id, text, metadata, distance in zip(
            ids,
            texts,
            metadatas,
            distances,
            strict=True,
        ):
            formatted.append({
                "chunk_id": chunk_id,
                "doc_id": metadata["doc_id"],
                "page_number": metadata["page_number"],
                "text": text,
                "distance": float(distance),
            })

    return formatted

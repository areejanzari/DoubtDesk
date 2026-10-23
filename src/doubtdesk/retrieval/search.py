import numpy as np
from sentence_transformers import SentenceTransformer
from chromadb.api.models.Collection import Collection

def embed_query(
        question: str,
        model: SentenceTransformer
) -> np.ndarray:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    return model.encode(
        question,
        normalize_embeddings=True,
    )


def search_chunks(
        collection: Collection,
        question_embedding: np.ndarray,
        doc_ids: list[str],
        top_k: int=5,
) -> dict:
    if not doc_ids:
        raise ValueError("Atleast one document ID is required.")

    if top_k <= 0:
        raise ValueError("top_k must be positive")

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=top_k,
        where={"doc_id": {"$in": doc_ids}},
        include=["documents","metadatas","distances"],
    )

    return results


def format_search_results(results: dict) -> list[dict]:
    formatted = []

    for chunk_id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
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

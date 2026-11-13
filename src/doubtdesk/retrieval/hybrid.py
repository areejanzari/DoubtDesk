import os
from collections import defaultdict
from doubtdesk.retrieval.search import embed_queries, format_search_results, search_chunks
from doubtdesk.retrieval.lexical_store import lexical_candidates


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_k: int = 8, k: int = 60):
    scores = defaultdict(float)
    candidates = {}
    for ranked in ranked_lists:
        seen = set()
        for rank, item in enumerate(ranked, start=1):
            key = item["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            scores[key] += 1.0 / (k + rank)
            if key not in candidates:
                candidates[key] = dict(item)
            elif "distance" in item:
                candidates[key]["distance"] = min(
                    candidates[key].get("distance", float("inf")), item["distance"]
                )
    result = sorted(candidates.values(), key=lambda item: (-scores[item["chunk_id"]], item["chunk_id"]))
    for item in result[:top_k]:
        item["rrf_score"] = scores[item["chunk_id"]]
        item.setdefault("distance", 1.0)
    return result[:top_k]


def hybrid_search(collection, embedding_model, questions: list[str], doc_ids: list[str], top_k: int | None = None, candidate_k: int = 12):
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "6")) if top_k is None else top_k
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("Retrieval top_k and candidate_k must be positive")
    if not doc_ids or not questions:
        return []
    dense_results = search_chunks(
        collection,
        embed_queries(questions, embedding_model),
        doc_ids,
        top_k=candidate_k,
    )
    dense_lists = []
    for ids, texts, metadata, distances in zip(
        dense_results["ids"],
        dense_results["documents"],
        dense_results["metadatas"],
        dense_results["distances"],
        strict=True,
    ):
        dense_lists.append(format_search_results({
            "ids": [ids],
            "documents": [texts],
            "metadatas": [metadata],
            "distances": [distances],
        }))
    sparse_lists = lexical_candidates(questions, doc_ids, candidate_k)
    return reciprocal_rank_fusion(dense_lists + sparse_lists, top_k=top_k)


def forward_evidence_chunks(ranked_chunks: list[dict], max_chunks: int | None = None) -> list[dict]:
    limit = int(os.getenv("EVIDENCE_FORWARD_TOP_K", "4")) if max_chunks is None else max_chunks
    if limit <= 0:
        raise ValueError("EVIDENCE_FORWARD_TOP_K must be positive")
    return ranked_chunks[:limit]

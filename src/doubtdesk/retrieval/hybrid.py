from doubtdesk.config import runtime_settings
import asyncio
import re
from collections import defaultdict
from doubtdesk.retrieval.search import embed_queries, search_chunks
from doubtdesk.retrieval.lexical_store import lexical_candidates
from doubtdesk.retrieval.rerank import rerank_chunks

CANDIDATE_K = 20
RERANK_POOL = 20
NEIGHBOUR_SOURCES = 3
MAX_NEIGHBOURS = 3
DUPLICATE_JACCARD = 0.6
DENSE_WEIGHT = 1.0
LEXICAL_WEIGHT = 1.0
_WORD = re.compile(r"[^\W_]+", re.UNICODE)


def reciprocal_rank_fusion(ranked_lists: list[list[dict]], top_k: int = 8, k: int = 60, weights: list[float] | None = None):
    scores = defaultdict(float)
    candidates = {}
    for position, ranked in enumerate(ranked_lists):
        weight = 1.0 if weights is None else weights[position]
        seen = set()
        for rank, item in enumerate(ranked, start=1):
            key = item["chunk_id"]
            if key in seen:
                continue
            seen.add(key)
            scores[key] += weight / (k + rank)
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


def retrieval_queries(question: str | None, paraphrases: list[str] | None) -> list[str]:
    queries = [query.strip() for query in [question or "", *(paraphrases or [])] if query and query.strip()]
    return list(dict.fromkeys(queries))


def _fuse(dense_lists, sparse_lists, top_k):
    weights = [DENSE_WEIGHT] * len(dense_lists) + [LEXICAL_WEIGHT] * len(sparse_lists)
    return reciprocal_rank_fusion(dense_lists + sparse_lists, top_k=top_k, weights=weights)


def hybrid_search(embedding_model, questions: list[str], doc_ids: list[str], top_k: int | None = None, candidate_k: int = CANDIDATE_K):
    top_k = runtime_settings().retrieval_top_k if top_k is None else top_k
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("Retrieval top_k and candidate_k must be positive")
    if not doc_ids or not questions:
        return []
    dense_lists = search_chunks(
        embed_queries(questions, embedding_model),
        doc_ids,
        top_k=candidate_k,
    )
    sparse_lists = lexical_candidates(questions, doc_ids, candidate_k)
    return _fuse(dense_lists, sparse_lists, top_k)


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.casefold()))


def drop_near_duplicates(chunks: list[dict], threshold: float = DUPLICATE_JACCARD) -> list[dict]:
    kept = []
    seen_words = []
    for chunk in chunks:
        words = _words(chunk["text"])
        duplicate = False
        for other, other_words in zip(kept, seen_words):
            if other["doc_id"] != chunk["doc_id"] or other["page_number"] != chunk["page_number"]:
                continue
            union = words | other_words
            if union and len(words & other_words) / len(union) >= threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(chunk)
            seen_words.append(words)
    return kept


def forward_evidence_chunks(ranked_chunks: list[dict], max_chunks: int | None = None) -> list[dict]:
    limit = runtime_settings().evidence_forward_top_k if max_chunks is None else max_chunks
    if limit <= 0:
        raise ValueError("EVIDENCE_FORWARD_TOP_K must be positive")
    return drop_near_duplicates(list(ranked_chunks))[:limit]


def pin_referenced_pages(ranked: list[dict], question: str, doc_ids: list[str], reranker=None,
                         find_pages=None, fetch=None) -> list[dict]:
    from doubtdesk.ingestion.figures import question_refs
    refs = question_refs(question or "")
    if not refs or not doc_ids:
        return ranked
    if find_pages is None or fetch is None:
        from doubtdesk.retrieval.vector_store import chunks_for_pages, pages_with_figure_refs
        find_pages = find_pages or pages_with_figure_refs
        fetch = fetch or chunks_for_pages
    try:
        pages = list(find_pages(doc_ids, refs))
    except Exception:
        return ranked
    if not pages:
        return ranked
    wanted = set(pages)
    pinned = [chunk for chunk in ranked if (chunk["doc_id"], int(chunk["page_number"])) in wanted]
    covered = {(chunk["doc_id"], int(chunk["page_number"])) for chunk in pinned}
    missing = [page for page in pages if page not in covered]
    if missing:
        try:
            fetched = fetch(missing)
        except Exception:
            fetched = []
        if fetched and reranker is not None:
            fetched = rerank_chunks(reranker, question, fetched)
        pinned.extend(fetched)
    pinned_ids = {chunk["chunk_id"] for chunk in pinned}
    for chunk in pinned:
        chunk["pinned_reference"] = True
    return pinned + [chunk for chunk in ranked if chunk["chunk_id"] not in pinned_ids]


def with_neighbours(forwarded: list[dict], doc_ids: list[str], lookup=None) -> list[dict]:
    if not forwarded:
        return forwarded
    if lookup is None:
        from doubtdesk.retrieval.vector_store import neighbour_chunks
        lookup = neighbour_chunks
    sources = forwarded[:NEIGHBOUR_SOURCES]
    present = {chunk["chunk_id"] for chunk in forwarded}
    try:
        found = list(lookup(sources, doc_ids))
    except Exception:
        return forwarded
    order = {(chunk["doc_id"], int(chunk["page_number"]), int(chunk.get("chunk_index", -1))): index
             for index, chunk in enumerate(sources)}
    found.sort(key=lambda item: (order.get(item.get("neighbour_of"), len(order)), item["page_number"], item["chunk_index"]))
    extras = []
    for item in found:
        if item["chunk_id"] in present or len(extras) >= MAX_NEIGHBOURS:
            continue
        present.add(item["chunk_id"])
        extras.append({key: value for key, value in item.items() if key != "neighbour_of"} | {"context_only": True})
    return forwarded + extras


def retrieve_ranked_evidence(embedding_model, queries: list[str], doc_ids: list[str], *, question: str | None = None,
                             reranker=None, pool_k: int = RERANK_POOL, neighbours: bool = True):
    queries = retrieval_queries(None, queries)
    if not queries or not doc_ids:
        return [], []
    ranked = hybrid_search(embedding_model, queries, doc_ids, top_k=pool_k)
    ranked = rerank_chunks(reranker, question or queries[0], ranked)
    ranked = pin_referenced_pages(ranked, question or queries[0], doc_ids, reranker)
    forwarded = forward_evidence_chunks(ranked)
    if neighbours:
        forwarded = with_neighbours(forwarded, doc_ids)
    return ranked, forwarded


async def ahybrid_search(embedding_model, questions: list[str], doc_ids: list[str], top_k: int | None = None, candidate_k: int = CANDIDATE_K):
    top_k = runtime_settings().retrieval_top_k if top_k is None else top_k
    if top_k <= 0 or candidate_k <= 0:
        raise ValueError("Retrieval top_k and candidate_k must be positive")
    if not doc_ids or not questions:
        return []

    async def dense():
        embeddings = await asyncio.to_thread(embed_queries, questions, embedding_model)
        return await asyncio.to_thread(search_chunks, embeddings, doc_ids, top_k=candidate_k)

    dense_lists, sparse_lists = await asyncio.gather(
        dense(),
        asyncio.to_thread(lexical_candidates, questions, doc_ids, candidate_k),
    )
    return _fuse(dense_lists, sparse_lists, top_k)


async def aretrieve_ranked_evidence(embedding_model, queries: list[str], doc_ids: list[str], *, question: str | None = None,
                                    reranker=None, pool_k: int = RERANK_POOL, neighbours: bool = True):
    queries = retrieval_queries(None, queries)
    if not queries or not doc_ids:
        return [], []
    ranked = await ahybrid_search(embedding_model, queries, doc_ids, top_k=pool_k)
    ranked = await asyncio.to_thread(rerank_chunks, reranker, question or queries[0], ranked)
    ranked = await asyncio.to_thread(pin_referenced_pages, ranked, question or queries[0], doc_ids, reranker)
    forwarded = forward_evidence_chunks(ranked)
    if neighbours:
        forwarded = await asyncio.to_thread(with_neighbours, forwarded, doc_ids)
    return ranked, forwarded

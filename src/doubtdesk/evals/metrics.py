HIT_CUTOFFS = (1, 4, 8, 20)


def ranked_pages(chunks: list[dict]) -> list[tuple[str, int]]:
    return list(dict.fromkeys((chunk["doc_id"], int(chunk["page_number"])) for chunk in chunks))


def page_metrics(chunks: list[dict], expected: set[tuple[str, int]]) -> dict | None:
    if not expected:
        return None
    pages = ranked_pages(chunks)
    rank = next((index for index, page in enumerate(pages, start=1) if page in expected), None)
    return {
        "first_relevant_rank": rank,
        "reciprocal_rank": 1.0 / rank if rank else 0.0,
        **{f"hit@{cutoff}": bool(rank and rank <= cutoff) for cutoff in HIT_CUTOFFS},
        "ranked_pages": [{"doc_id": doc_id, "page_number": number} for doc_id, number in pages[:20]],
    }


def summarize(records: list[dict]) -> dict:
    scored = [record["page_metrics"] for record in records if record.get("page_metrics")]
    if not scored:
        return {"questions": 0}
    count = len(scored)
    return {
        "questions": count,
        "mrr": round(sum(item["reciprocal_rank"] for item in scored) / count, 4),
        **{f"hit@{cutoff}": round(sum(item[f"hit@{cutoff}"] for item in scored) / count, 4) for cutoff in HIT_CUTOFFS},
    }

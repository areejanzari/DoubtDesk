import re
from doubtdesk.db import connection


STOPWORDS = frozenset("a an the is are am was were be been being of to in on at for from by with as and or what which who whom how why when where does do did can could would should will may might has have had this that these those it its you your i me my explain describe tell give".split())

TEXT_SEARCH_CONFIGS = (("english", "tsv_en"), ("simple", "tsv"))

LEXICAL_QUERY = """
WITH terms AS (
    SELECT phraseto_tsquery('{config}', word) AS query
    FROM unnest(%(words)s::text[]) AS word
),
usable AS (
    SELECT query FROM terms WHERE numnode(query) > 0
),
total AS (
    SELECT count(*) AS n FROM chunks WHERE doc_id = ANY(%(doc_ids)s)
),
weighted AS (
    SELECT usable.query,
           ln(1 + (total.n - matches.n + 0.5) / (matches.n + 0.5))::float8 AS idf
    FROM usable
    CROSS JOIN total
    CROSS JOIN LATERAL (
        SELECT count(*) AS n FROM chunks
        WHERE doc_id = ANY(%(doc_ids)s) AND {column} @@ usable.query
    ) AS matches
)
SELECT c.chunk_id, c.doc_id, c.page_number, c.text, c.chunk_index
FROM chunks AS c
JOIN weighted AS w ON c.{column} @@ w.query
WHERE c.doc_id = ANY(%(doc_ids)s)
GROUP BY c.chunk_id
ORDER BY sum(w.idf * ts_rank(c.{column}, w.query, 1)::float8) DESC, c.chunk_id
LIMIT %(top_k)s
"""

QUERIES = {config: LEXICAL_QUERY.format(config=config, column=column) for config, column in TEXT_SEARCH_CONFIGS}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:[.+\-/^]+[^\W_]+)*", text.casefold(), flags=re.UNICODE)


def query_words(question: str) -> list[str]:
    words = list(dict.fromkeys(tokenize(question)))
    significant = [word for word in words if word not in STOPWORDS]
    return significant or words


def lexical_candidates(
    questions: list[str],
    doc_ids: list[str],
    top_k: int = 12,
    configs: tuple[str, ...] = ("english", "simple"),
) -> list[list[dict]]:
    if not questions or not doc_ids:
        return []
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    result = []
    with connection() as conn:
        for question in questions:
            words = query_words(question)
            if not words:
                result.append([])
                continue
            per_config = []
            for config in configs:
                rows = conn.execute(
                    QUERIES[config],
                    {"words": words, "doc_ids": list(doc_ids), "top_k": top_k},
                ).fetchall()
                per_config.append([
                    {
                        "chunk_id": row[0],
                        "doc_id": row[1],
                        "page_number": int(row[2]),
                        "text": row[3],
                        "chunk_index": int(row[4]),
                    }
                    for row in rows
                ])
            result.append(_fuse_configs(per_config, top_k))
    return result


def _fuse_configs(lists: list[list[dict]], top_k: int, k: int = 60) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked in lists:
        for rank, item in enumerate(ranked, start=1):
            scores[item["chunk_id"]] = scores.get(item["chunk_id"], 0.0) + 1.0 / (k + rank)
            items.setdefault(item["chunk_id"], item)
    ordered = sorted(items, key=lambda key: (-scores[key], key))
    return [items[key] for key in ordered[:top_k]]

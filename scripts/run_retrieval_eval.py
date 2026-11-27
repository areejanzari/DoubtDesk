import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from doubtdesk.db import connection, initialize_database
from doubtdesk.evals.metrics import page_metrics, summarize
from doubtdesk.evals.schemas import GoldenQuestion
from doubtdesk.ingestion.embeddings import load_embedding_model
from doubtdesk.ingestion.hashing import calculate_sha256
from doubtdesk.ingestion.registry import get_document_by_hash
from doubtdesk.ingestion.service import build_document, process_pdf
from doubtdesk.retrieval.hybrid import retrieve_ranked_evidence
from doubtdesk.retrieval.rerank import load_reranker

EVAL_USER = "retrieval-eval"


def ensure_documents(paths: list[Path], embedding_model, reingest: bool) -> dict[str, str]:
    doc_ids = {}
    for path in paths:
        existing = get_document_by_hash(calculate_sha256(path))
        if existing and existing["status"] == "completed" and reingest:
            build_document(existing["doc_id"], path, embedding_model)
            doc_id = existing["doc_id"]
        else:
            doc_id = process_pdf(path, user_id=EVAL_USER, embedding_model=embedding_model)["doc_id"]
        with connection() as conn:
            conn.execute(
                "INSERT INTO user_documents (user_id, doc_id, filename, attached_at) VALUES (%s, %s, %s, 0) "
                "ON CONFLICT DO NOTHING", (EVAL_USER, doc_id, path.name),
            )
        doc_ids[path.name] = doc_id
    return doc_ids


async def paraphrase_queries(questions: list[str]) -> list[list[str]]:
    from openai import AsyncOpenAI
    from doubtdesk.config import runtime_settings
    from doubtdesk.guardrails.paraphrasing import aresolve_and_paraphrase
    config = runtime_settings()
    client = AsyncOpenAI(api_key=config.openai_api_key, max_retries=2)
    try:
        result = []
        for question in questions:
            resolved = await aresolve_and_paraphrase(client=client, model_name=config.paraphrase_model or config.openai_model,
                                                     question=question, history=[])
            result.append(list(dict.fromkeys([resolved.resolved_question, *resolved.questions])))
        return result
    finally:
        await client.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("eval_file", type=Path, nargs="?", default=Path("examples/retrieval_evals.json"))
    parser.add_argument("--output", type=Path, default=Path("data/retrieval_eval_report.json"))
    parser.add_argument("--paraphrase", action="store_true")
    parser.add_argument("--reingest", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    spec = json.loads(args.eval_file.read_text(encoding="utf-8"))
    questions = [GoldenQuestion.model_validate(item) for item in spec["questions"]]
    initialize_database()
    embedding_model = load_embedding_model()
    reranker = None if args.no_rerank else load_reranker()
    doc_ids = ensure_documents([Path(path) for path in spec["documents"]], embedding_model, args.reingest)
    all_doc_ids = list(doc_ids.values())

    query_sets = (asyncio.run(paraphrase_queries([item.question for item in questions])) if args.paraphrase
                  else [[item.question] for item in questions])
    records = []
    by_category = defaultdict(list)
    for golden, queries in zip(questions, query_sets):
        ranked, forwarded = retrieve_ranked_evidence(embedding_model, queries, all_doc_ids,
                                                     question=queries[0], reranker=reranker)
        expected = {(doc_ids[page.filename] if page.filename else page.doc_id, page.page_number)
                    for page in golden.expected_pages}
        record = {
            "question": golden.question,
            "category": golden.category,
            "queries": queries,
            "page_metrics": page_metrics(ranked, expected),
            "forwarded_hit": bool(expected & {(chunk["doc_id"], int(chunk["page_number"])) for chunk in forwarded}) if expected else None,
            "forwarded_pages": [[chunk["doc_id"][:8], chunk["page_number"], round(chunk.get("rerank_score", 0.0), 3)] for chunk in forwarded],
            "top_rerank_score": max((chunk["rerank_score"] for chunk in forwarded if "rerank_score" in chunk), default=None),
        }
        records.append(record)
        by_category[golden.category or "uncategorised"].append(record)

    scored = [record for record in records if record["forwarded_hit"] is not None]
    report = {
        "overall": {**summarize(records),
                    "forwarded_hit": round(sum(record["forwarded_hit"] for record in scored) / max(len(scored), 1), 4)},
        "by_category": {name: summarize(items) for name, items in sorted(by_category.items())},
        "unanswerable_top_scores": [
            {"question": record["question"], "category": record["category"], "top_rerank_score": record["top_rerank_score"]}
            for record in records if record["page_metrics"] is None
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "by_category": report["by_category"],
                      "unanswerable_top_scores": report["unanswerable_top_scores"]}, indent=2))


if __name__ == "__main__":
    main()

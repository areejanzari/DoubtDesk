import asyncio
import logging
from uuid import uuid4
from doubtdesk.db import connection
from doubtdesk.evals.store import update_run
from doubtdesk.guardrails.paraphrasing import aresolve_and_paraphrase
from doubtdesk.retrieval.hybrid import aretrieve_ranked_evidence, retrieval_queries
from doubtdesk.evals.metrics import page_metrics
from doubtdesk.observability.tracing import trace_run, graph_config

logger = logging.getLogger(__name__)


def get_evaluation_owner(doc_ids: list[str]) -> str | None:
    with connection() as conn:
        row = conn.execute("""
            SELECT user_id FROM user_documents
            WHERE doc_id = ANY(%s)
            GROUP BY user_id HAVING COUNT(DISTINCT doc_id) = %s
            ORDER BY user_id
            LIMIT 1
        """, (list(doc_ids), len(doc_ids))).fetchone()
    return row[0] if row else None


async def evaluate(run_id, payload, app):
    judge = None
    records = []
    await asyncio.to_thread(update_run, run_id, status="running")
    try:
        from doubtdesk.evals.judge import RagasJudge
        judge = RagasJudge(embedding_model=getattr(app.state, "embedding_model", None))
        owner = await asyncio.to_thread(get_evaluation_owner, payload.doc_ids) if payload.mode == "full" else None
        if payload.mode == "full" and not owner:
            raise ValueError("Full evaluation requires one user with access to all selected document IDs")
        for index, golden in enumerate(payload.questions, start=1):
            try:
                with trace_run("doubtdesk.admin_eval", user_id=owner):
                    if payload.mode == "full":
                        result = await app.state.graph.ainvoke({
                            "user_id": owner,
                            "question": golden.question,
                            "doc_ids": payload.doc_ids,
                            "request_id": str(uuid4()),
                        }, config=graph_config())
                        chunks = result.get("retrieved_chunks", [])
                        answer = result.get("answer") if result.get("status") == "grounded" else None
                        status = result.get("status")
                    else:
                        resolved = await aresolve_and_paraphrase(
                            client=getattr(app.state, "paraphrase_client", app.state.llm_client),
                            model_name=getattr(app.state, "paraphrase_model_name", app.state.model_name),
                            question=golden.question,
                            history=[],
                            request_id=str(uuid4()),
                        )
                        ranked, chunks = await aretrieve_ranked_evidence(
                            app.state.embedding_model,
                            retrieval_queries(resolved.resolved_question, resolved.questions),
                            payload.doc_ids,
                            question=resolved.resolved_question,
                            reranker=getattr(app.state, "reranker", None),
                        )
                        answer = None
                        status = "retrieval_only"
                    expected = {(page.doc_id, page.page_number) for page in golden.expected_pages if page.doc_id}
                    ranking = ranked if payload.mode == "retrieval" else chunks
                    contexts = [chunk["text"] for chunk in chunks]
                    if not contexts:
                        raise ValueError("Retriever returned no text chunks")
                    evaluation = await judge.score(
                        question=golden.question,
                        contexts=contexts,
                        reference=golden.reference_answer,
                        requirements=golden.must_include,
                        answer=answer,
                        thresholds=payload.thresholds,
                    )
                    if payload.mode == "full" and answer is None:
                        for name in ("faithfulness", "answer_relevancy", "answer_correctness"):
                            evaluation["scores"][name] = None
                            evaluation["passed"][name] = False
                            evaluation["metric_errors"][name] = (
                                f"No grounded generated answer (status={status})"
                            )
                        evaluation["all_passed"] = False
                    records.append({
                        "index": index,
                        "question": golden.question,
                        "reference_answer": golden.reference_answer,
                        "must_include": golden.must_include,
                        "answer": answer,
                        "generation_status": status,
                        "judge_failure_reason": result.get("judge_failure_reason") if payload.mode == "full" else None,
                        "judge_reviews": result.get("judge_reviews", []) if payload.mode == "full" else [],
                        "judge_attempts": result.get("judge_attempts", 0) if payload.mode == "full" else 0,
                        "evidence_tier": result.get("evidence_tier") if payload.mode == "full" else None,
                        "evidence_average": result.get("evidence_average") if payload.mode == "full" else None,
                        "page_metrics": page_metrics(ranking, expected),
                        "retrieved_contexts": [
                            {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                             "page_number": chunk["page_number"], "rrf_score": chunk.get("rrf_score"),
                             "rerank_score": chunk.get("rerank_score"), "text": chunk["text"]}
                            for chunk in chunks
                        ],
                        **evaluation,
                    })
            except Exception as exc:
                logger.exception("Evaluation item %s failed", index)
                records.append({
                    "index": index,
                    "question": golden.question,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                    "all_passed": False,
                })
            await asyncio.to_thread(update_run, run_id, progress=index, result=records)
        await asyncio.to_thread(update_run, run_id, status="completed", result=records)
    except Exception as exc:
        logger.exception("Evaluation job failed")
        await asyncio.to_thread(update_run, run_id, status="failed", result=records, error=str(exc)[:500])
    finally:
        if judge is not None:
            await judge.close()


async def run_evaluation_job(run_id, payload, app):
    await evaluate(run_id, payload, app)

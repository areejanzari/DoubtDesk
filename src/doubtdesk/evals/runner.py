import asyncio
import logging
import os
import sqlite3
from contextlib import closing
from uuid import uuid4
from doubtdesk.evals.judge import RagasJudge
from doubtdesk.evals.store import update_run
from doubtdesk.guardrails.paraphrasing import resolve_and_paraphrase
from doubtdesk.retrieval.hybrid import hybrid_search, forward_evidence_chunks
from doubtdesk.observability.tracing import trace_run, graph_config

logger = logging.getLogger(__name__)


def get_evaluation_owner(doc_ids: list[str]) -> str | None:
    placeholders = ",".join("?" for _ in doc_ids)
    with closing(sqlite3.connect("data/documents.db")) as conn:
        row = conn.execute(f"""
            SELECT user_id FROM user_documents
            WHERE doc_id IN ({placeholders})
            GROUP BY user_id HAVING COUNT(DISTINCT doc_id) = ?
            LIMIT 1
        """, (*doc_ids, len(doc_ids))).fetchone()
    return row[0] if row else None


async def evaluate(run_id, payload, app):
    judge = None
    records = []
    update_run(run_id, status="running")
    try:
        judge = RagasJudge()
        owner = get_evaluation_owner(payload.doc_ids) if payload.mode == "full" else None
        if payload.mode == "full" and not owner:
            raise ValueError("Full evaluation requires one user with access to all selected document IDs")
        for index, golden in enumerate(payload.questions, start=1):
            try:
                with trace_run("doubtdesk.admin_eval", user_id=owner):
                    if payload.mode == "full":
                        result = app.state.graph.invoke({
                            "user_id": owner,
                            "question": golden.question,
                            "doc_ids": payload.doc_ids,
                            "request_id": str(uuid4()),
                        }, config=graph_config())
                        chunks = result.get("retrieved_chunks", [])
                        answer = result.get("answer") if result.get("status") == "grounded" else None
                        status = result.get("status")
                    else:
                        paraphrases = resolve_and_paraphrase(
                            client=app.state.llm_client,
                            model_name=getattr(app.state, "paraphrase_model_name", app.state.model_name),
                            question=golden.question,
                            history=[],
                            request_id=str(uuid4()),
                        ).questions
                        chunks = forward_evidence_chunks(hybrid_search(
                            app.state.collection,
                            app.state.embedding_model,
                            paraphrases,
                            payload.doc_ids,
                        ))
                        answer = None
                        status = "retrieval_only"
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
                        "retrieved_contexts": [
                            {"chunk_id": chunk["chunk_id"], "doc_id": chunk["doc_id"],
                             "page_number": chunk["page_number"], "rrf_score": chunk["rrf_score"],
                             "text": chunk["text"]}
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
            update_run(run_id, progress=index, result=records)
        update_run(run_id, status="completed", result=records)
    except Exception as exc:
        logger.exception("Evaluation job failed")
        update_run(run_id, status="failed", result=records, error=str(exc)[:500])
    finally:
        if judge is not None:
            await judge.close()


def run_evaluation_job(run_id, payload, app):
    asyncio.run(evaluate(run_id, payload, app))

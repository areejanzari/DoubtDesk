import os
import sqlite3
from contextlib import closing
from uuid import uuid4, UUID
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from doubtdesk.analytics.auth import require_admin_key
from doubtdesk.ingestion.registry import validate_document_ids
from doubtdesk.evals.schemas import EvalsRequest
from doubtdesk.evals.store import create_run, get_run
from doubtdesk.evals.runner import run_evaluation_job

router = APIRouter(
    prefix="/admin/evals", tags=["Admin Evals"],
    dependencies=[Depends(require_admin_key)],
)


@router.post("", status_code=202)
def create_evaluation(payload: EvalsRequest, background: BackgroundTasks, request: Request):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=503, detail="GEMINI_API_KEY not configured for RAGAS judge")
    with closing(sqlite3.connect("data/documents.db")) as conn:
        try:
            validate_document_ids(conn, payload.doc_ids)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    run_id = str(uuid4())
    create_run(run_id, payload.mode, len(payload.questions))
    background.add_task(run_evaluation_job, run_id, payload, request.app)
    return {"run_id": run_id, "status": "queued", "results_url": f"/admin/evals/{run_id}"}


@router.get("/{run_id}")
def retrieve_evaluation(run_id: UUID):
    result = get_run(str(run_id))
    if result is None:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return result

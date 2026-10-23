import sqlite3
from fastapi import APIRouter, HTTPException, Request
from doubtdesk.api.schemas import AskRequest, AskResponse
from doubtdesk.ingestion.registry import validate_document_ids

router = APIRouter()

@router.get('health')
def health():
    return { "status": "ok"}


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request):
    conn = sqlite3.connect("data/documents.db")

    try:
        validate_document_ids(conn, payload.doc_ids)

        result = request.app.state.graph.invoke({
            "question": payload.question,
            "doc_ids": payload.doc_ids,
        })

        return AskResponse.model_validate(result)

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    finally:
        conn.close()

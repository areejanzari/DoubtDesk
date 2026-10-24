import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from doubtdesk.ingestion.service import process_pdf
from doubtdesk.api.schemas import AskRequest, AskResponse,DocumentInfo, UploadResponse
from doubtdesk.ingestion.registry import (
    validate_document_ids,
    list_documents,
    get_document_names,
    update_document_filename,)

router = APIRouter()

@router.get('/health')
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

@router.get('/documents',response_model=list[DocumentInfo])
def get_documents():
    with sqlite3.connect("data/documents.db") as conn:
        documents = list_documents(conn)

    return documents

@router.post('/documents/upload',response_model=UploadResponse)
def upload_document(file: UploadFile= File(...)):
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only pdf files are supported.",
        )

    try:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / filename

            with pdf_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):    
                    destination.write(chunk)                 

            if pdf_path.stat().st_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded PDF is empty.",
                )

            result = process_pdf(pdf_path)

            if result["status"] == "cached":
                        update_document_filename(
                            doc_id=result["doc_id"],
                            filename=filename,
                        )

        
            with sqlite3.connect("data/documents.db") as conn:
                document_names = get_document_names(
                    conn,
                    [result["doc_id"]],
                )

            return UploadResponse(
                **result,
                filename=document_names[result["doc_id"]],
            )


    finally:
        file.file.close()



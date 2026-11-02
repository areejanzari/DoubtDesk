import sqlite3
import pymupdf
from contextlib import closing
from uuid import UUID, uuid4
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Depends
from fastapi.responses import FileResponse
from doubtdesk.ingestion.service import process_pdf
from doubtdesk.auth.dependencies import require_current_user
from doubtdesk.api.schemas import AskRequest, AskResponse,DocumentInfo, UploadResponse
from doubtdesk.memory.store import save_exchange,claim_or_check_conversation
from doubtdesk.ingestion.registry import (
    validate_user_document_ids,
    list_documents,
    get_document_names,
    user_can_access_document,
)

router = APIRouter()

@router.get('/health')
def health():
    return { "status": "ok"}



@router.post("/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    request: Request,
    user: dict = Depends(require_current_user),
):
    user_id = user["user_id"]
    conversation_id = payload.conversation_id or uuid4()
    request_id = str(uuid4())

    with closing(sqlite3.connect("data/documents.db")) as conn:
        try:
            validate_user_document_ids(
                conn,
                user_id=user_id,
                doc_ids=payload.doc_ids,
            )
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

    allowed = claim_or_check_conversation(
        conversation_id=str(conversation_id),
        user_id=user_id,
    )

    if not allowed:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found.",
        )

    try:
        result = request.app.state.graph.invoke({
            "user_id": user_id,
            "question": payload.question,
            "doc_ids": payload.doc_ids,
            "conversation_id": str(conversation_id),
            "request_id": request_id,
        })

        response = AskResponse.model_validate({
            **result,
            "conversation_id": conversation_id,
        })

        save_exchange(
            conversation_id=str(conversation_id),
            question=payload.question,
            answer=response.answer,
            doc_ids=payload.doc_ids,
            status=response.status,
        )

        return response

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc



@router.get("/documents", response_model=list[DocumentInfo])
def get_documents(
    user: dict = Depends(require_current_user),
):
    with closing(sqlite3.connect("data/documents.db")) as conn:
        documents = list_documents(
            conn,
            user_id=user["user_id"],
        )

    return documents



@router.post("/documents/upload", response_model=UploadResponse)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    user: dict = Depends(require_current_user),
):
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only pdf files are supported.",
        )

    try:
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / filename

            MAX_UPLOAD_SIZE = 25 * 1024 * 1024
            total_size = 0

            with pdf_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    total_size += len(chunk)

                    if total_size > MAX_UPLOAD_SIZE:
                        raise HTTPException(
                            status_code=413,
                            detail="PDF exceeds the 25 MB upload limit.",
                        )

                    destination.write(chunk)

            if pdf_path.stat().st_size == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Uploaded PDF is empty.",
                )

            with pdf_path.open("rb") as pdf:
                signature = pdf.read(5)

            if signature != b"%PDF-":
                raise HTTPException(
                    status_code=400,
                    detail="Invalid PDF file content.",
                )

            try:
                with pdf_path.open("rb") as pdf:
                    pdf_bytes = pdf.read()

                with pymupdf.open(
                    stream=pdf_bytes,
                    filetype="pdf",
                ) as document:
                    if document.page_count == 0:
                        raise HTTPException(
                            status_code=400,
                            detail="PDF contains no pages.",
                        )

            except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail="PDF is corrupted or unreadable.",
                ) from exc

            result = process_pdf(
                pdf_path,
                user_id=user["user_id"],
                embedding_model=request.app.state.embedding_model,
                collection=request.app.state.collection,
            )

            return UploadResponse(
                **result,
                filename=filename,
            )

    finally:
        file.file.close()


@router.get("/documents/{doc_id}/pages/{page_number}/image")
def get_page_image(
    doc_id: str,
    page_number: int,
    user: dict = Depends(require_current_user),
):
    try:
        UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID.",
        )

    if page_number < 1:
        raise HTTPException(
            status_code=400,
            detail="Page number must be greater than zero.",
        )

    with closing(sqlite3.connect("data/documents.db")) as conn:
        allowed = user_can_access_document(
            conn,
            user_id=user["user_id"],
            doc_id=doc_id,
            page_number=page_number,
        )

    if not allowed:
        raise HTTPException(
            status_code=404,
            detail="Document or page not found.",
        )

    image_dir = Path("data/page_images") / doc_id
    image_path = image_dir / f"page_{page_number:03d}.png"

    if not image_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Page image not found.",
        )

    return FileResponse(
        path=image_path,
        media_type="image/png",
    )

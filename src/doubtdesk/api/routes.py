from doubtdesk.config import runtime_settings
from doubtdesk.api.stream_context import ContextStreamingResponse
import asyncio
import mmap
import shutil
import pymupdf
from uuid import UUID, uuid4
from pathlib import Path
from tempfile import TemporaryDirectory
from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Depends, Response
from doubtdesk.storage.images import read_page_image
from doubtdesk.db import connection
from doubtdesk.config import settings
from doubtdesk.ingestion.service import start_ingestion, run_ingestion
from doubtdesk.ingestion.jobs import submit as submit_ingestion
from doubtdesk.ingestion.registry import DocumentProcessing
from doubtdesk.auth.dependencies import require_current_user
from doubtdesk.api.schemas import AskRequest, AskResponse,DocumentInfo, DocumentStatus, UploadResponse, UsageResponse
from doubtdesk.quotas import QuotaExceeded, acquire_ask_slot, release_ask_slot, get_user_usage
from doubtdesk.memory.store import save_exchange, claim_or_check_conversation, ConversationLimitReached, delete_empty_conversation
from doubtdesk.memory.vector import add_exchange_memory
from doubtdesk.observability.tracing import trace_run, graph_config, trace_stage, update_observation
from doubtdesk.api.streaming import sse, astream_answer, finalize_stream_result, ainvoke_graph, RESULT
import logging
logger = logging.getLogger(__name__)
from doubtdesk.ingestion.registry import (
    validate_user_document_ids,
    list_documents,
    user_can_access_document,
    get_user_document_status,
)

router = APIRouter()
MAX_UPLOAD_SIZE = 25 * 1024 * 1024

@router.get('/health')
def health():
    return { "status": "ok"}



QUOTA_MESSAGES = {
    "concurrency": "You already have a question in progress. Wait for it to finish and try again.",
    "questions": "You have reached your daily question limit for your plan.",
    "tokens": "You have reached your daily usage limit for your plan.",
}


def _user_plan(user):
    return user.get("plan") or "free"


def _prepare_request(payload, user):
    user_id = user["user_id"]
    conversation_id = payload.conversation_id or uuid4()
    request_id = str(uuid4())

    with connection() as conn:
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

    try:
        acquire_ask_slot(user_id, _user_plan(user), request_id)
    except QuotaExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=QUOTA_MESSAGES[exc.reason],
            headers={"Retry-After": str(exc.retry_after)},
        ) from exc

    try:
        try:
            allowed = claim_or_check_conversation(
                conversation_id=str(conversation_id),
                user_id=user_id,
            )
        except ConversationLimitReached as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        if not allowed:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found.",
            )
    except BaseException:
        release_ask_slot(request_id, "failed")
        raise
    return user_id, conversation_id, request_id


def _finish_request(request_id, status):
    try:
        release_ask_slot(request_id, status)
    except Exception:
        logger.exception("Failed to release ask quota slot")


def _cleanup_automatic_chat(payload, conversation_id, user_id):
    if payload.conversation_id is None:
        try:
            delete_empty_conversation(str(conversation_id), user_id)
        except Exception:
            logger.exception("Failed to clean up unused automatic conversation")


@router.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    user: dict = Depends(require_current_user),
):
    user_id, conversation_id, request_id = await asyncio.to_thread(_prepare_request, payload, user)
    outcome = "failed"

    try:
        with trace_run("doubtdesk.ask", user_id=user_id, session_id=str(conversation_id)):
            result = await ainvoke_graph(request.app.state.graph, {
                "user_id": user_id,
                "question": payload.question,
                "doc_ids": payload.doc_ids,
                "conversation_id": str(conversation_id),
                "request_id": request_id,
            }, graph_config())

        response = AskResponse.model_validate({
            **result,
            "conversation_id": conversation_id,
        })

        await asyncio.to_thread(
            _save_exchange_and_memory,
            request.app,
            user_id,
            conversation_id,
            payload.question,
            payload.doc_ids,
            response,
            memory_eligible=not result.get("document_insufficient", False),
        )

        outcome = "completed"
        return response

    except Exception as exc:
        logger.exception("Answer generation failed")
        await asyncio.to_thread(_cleanup_automatic_chat, payload, conversation_id, user_id)
        raise HTTPException(status_code=500, detail="Answer generation failed; try again.") from exc
    finally:
        await asyncio.to_thread(_finish_request, request_id, outcome)



@router.get("/usage", response_model=UsageResponse)
def get_usage(
    user: dict = Depends(require_current_user),
):
    return get_user_usage(user["user_id"], _user_plan(user))


@router.get("/documents", response_model=list[DocumentInfo])
def get_documents(
    user: dict = Depends(require_current_user),
):
    with connection() as conn:
        documents = list_documents(
            conn,
            user_id=user["user_id"],
        )

    return documents



@router.post("/documents/upload", response_model=UploadResponse)
def upload_document(
    request: Request,
    response: Response,
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

            total_size = 0

            with pdf_path.open("wb") as destination:
                while chunk := file.file.read(1024 * 1024):
                    if total_size == 0 and not chunk.startswith(b"%PDF-"):
                        raise HTTPException(status_code=400, detail="Invalid PDF file content.")
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

            try:
                with pdf_path.open("rb") as source:
                    with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
                        with memoryview(mapped) as view:
                            with pymupdf.open(stream=view, filetype="pdf") as document:
                                if not document.is_pdf:
                                    raise HTTPException(status_code=400, detail="Invalid PDF file content.")
                                if document.needs_pass:
                                    raise HTTPException(status_code=400, detail="Encrypted PDFs are not supported.")
                                if document.page_count == 0:
                                    raise HTTPException(status_code=400, detail="PDF contains no pages.")
                                max_pages = runtime_settings().max_pdf_pages
                                if document.page_count > max_pages:
                                    raise HTTPException(status_code=400, detail=f"PDF exceeds the {max_pages}-page limit.")
            except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError, OSError) as exc:
                raise HTTPException(status_code=400, detail="PDF is corrupted or unreadable.") from exc

            settings.upload_dir.mkdir(parents=True, exist_ok=True)
            stored_path = settings.upload_dir / f"{uuid4()}.pdf"
            shutil.move(pdf_path, stored_path)
            try:
                result = start_ingestion(
                    stored_path,
                    filename=filename,
                    user_id=user["user_id"],
                )
            except DocumentProcessing as exc:
                stored_path.unlink(missing_ok=True)
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception:
                stored_path.unlink(missing_ok=True)
                raise

            if result.pop("queued"):
                submit_ingestion(run_ingestion, result["doc_id"], stored_path, request.app.state.embedding_model)
                response.status_code = 202
            else:
                stored_path.unlink(missing_ok=True)

            return UploadResponse(
                **result,
                filename=filename,
            )

    finally:
        file.file.close()


@router.get("/documents/{doc_id}/status", response_model=DocumentStatus)
def get_document_status(
    doc_id: str,
    user: dict = Depends(require_current_user),
):
    try:
        UUID(doc_id)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid document ID.",
        )

    with connection() as conn:
        status = get_user_document_status(conn, user["user_id"], doc_id)

    if status is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return status


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

    with connection() as conn:
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

    try:
        image_bytes = read_page_image(doc_id, page_number)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Page image not found.",
        )

    return Response(
        content=image_bytes,
        media_type="image/png",
    )


def _save_exchange_and_memory(app, user_id, conversation_id, question, doc_ids, response, *, memory_eligible=True):
    exchange_id = save_exchange(
        conversation_id=str(conversation_id),
        question=question,
        answer=response.answer,
        doc_ids=doc_ids,
        status=response.status,
        user_id=user_id,
        formatted_answer=response.formatted_answer,
        sources=response.model_dump(mode="json")["sources"],
        memory_eligible=memory_eligible,
    )
    try:
        if memory_eligible and response.status not in ("out_of_scope", "verification_failed"):
            add_exchange_memory(
                exchange_id=exchange_id,
                question=question,
                answer=response.answer,
                model=app.state.embedding_model,
            )
    except Exception:
        logger.exception("Memory maintenance failed after answer was saved")


@router.post("/ask/stream")
async def ask_stream(
    payload: AskRequest,
    request: Request,
    user: dict = Depends(require_current_user),
):
    user_id, conversation_id, request_id = await asyncio.to_thread(_prepare_request, payload, user)

    app = request.app

    async def generate():
        yield sse("status", {"stage": "preparing", "conversation_id": str(conversation_id)})
        try:
            with trace_run(
                    "doubtdesk.ask.stream",
                    user_id=user_id,
                    session_id=str(conversation_id),
                ) as trace:
                update_observation(trace, input={"question": payload.question, "doc_ids": payload.doc_ids, "request_id": request_id})
                prepared = await ainvoke_graph(app.state.preparation_graph, {
                    "user_id": user_id,
                    "question": payload.question,
                    "doc_ids": payload.doc_ids,
                    "conversation_id": str(conversation_id),
                    "request_id": request_id,
                }, graph_config())
                yield sse("status", {"stage": "generating"})
                result = None
                answer_generator = astream_answer(
                    prepared, app.state.llm_client, app.state.model_name,
                    request_id, str(conversation_id),
                    judge=app.state.output_judge,
                )
                try:
                    async for event, data in answer_generator:
                        if event == RESULT:
                            result = data
                        else:
                            yield sse(event, data)
                finally:
                    await answer_generator.aclose()
                result = await asyncio.to_thread(finalize_stream_result, result, user_id=user_id, normalized=True)
                response = AskResponse.model_validate({**result, "conversation_id": conversation_id})
                with trace_stage("save_exchange", input={"status": response.status}):
                    await asyncio.to_thread(_save_exchange_and_memory, app, user_id, conversation_id, payload.question, payload.doc_ids, response, memory_eligible=not result.get("document_insufficient", False))
                await asyncio.to_thread(_finish_request, request_id, "completed")
                update_observation(trace, output={**response.model_dump(mode="json"), "request_id": request_id},
                                   metadata={"judge_attempts": result.get("judge_attempts"),
                                             "judge_regenerations": result.get("judge_regenerations"),
                                             "judge_failure_reason": result.get("judge_failure_reason")})
                yield sse("complete", response.model_dump(mode="json"))
        except GeneratorExit:
            raise
        except Exception as exc:
            logger.exception("Answer stream failed")
            await asyncio.to_thread(_finish_request, request_id, "failed")
            yield sse("error", {"type": type(exc).__name__, "message": "Answer stream failed; try again."})

    def on_close():
        _finish_request(request_id, "failed")
        _cleanup_automatic_chat(payload, conversation_id, user_id)

    return ContextStreamingResponse(
        generate(),
        on_close=on_close,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )

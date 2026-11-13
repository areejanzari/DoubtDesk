import logging
from uuid import UUID, uuid4
from fastapi import APIRouter, Depends, HTTPException, Request, status
from doubtdesk.auth.dependencies import require_current_user
from doubtdesk.conversations.schemas import (
    ConversationDetail,
    ConversationSummary,
    RenameConversationRequest,
)
from doubtdesk.memory.store import (
    ConversationLimitReached,
    claim_or_check_conversation,
    delete_user_conversation,
    get_user_conversation,
    list_user_conversations,
    rename_user_conversation,
)
from doubtdesk.memory.vector import delete_session_memories

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=list[ConversationSummary])
def list_conversations(user: dict = Depends(require_current_user)):
    return list_user_conversations(user["user_id"])


@router.post("", response_model=ConversationSummary, status_code=status.HTTP_201_CREATED)
def create_conversation(user: dict = Depends(require_current_user)):
    conversation_id = str(uuid4())
    try:
        claim_or_check_conversation(conversation_id, user["user_id"])
    except ConversationLimitReached as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return next(
        item for item in list_user_conversations(user["user_id"])
        if item["conversation_id"] == conversation_id
    )


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(
    conversation_id: UUID,
    user: dict = Depends(require_current_user),
):
    result = get_user_conversation(str(conversation_id), user["user_id"])
    if result is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return result


@router.patch("/{conversation_id}", response_model=ConversationSummary)
def rename_conversation(
    conversation_id: UUID,
    payload: RenameConversationRequest,
    user: dict = Depends(require_current_user),
):
    try:
        changed = rename_user_conversation(str(conversation_id), user["user_id"], payload.title)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not changed:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return next(
        item for item in list_user_conversations(user["user_id"])
        if item["conversation_id"] == str(conversation_id)
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: UUID,
    request: Request,
    user: dict = Depends(require_current_user),
):
    deleted = delete_user_conversation(str(conversation_id), user["user_id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    try:
        delete_session_memories(
            [str(conversation_id)], collection=request.app.state.memory_collection
        )
    except Exception:
        logger.exception("Failed to remove conversation vector memory after SQLite deletion")

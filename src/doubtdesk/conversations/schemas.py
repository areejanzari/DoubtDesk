from uuid import UUID
from pydantic import BaseModel, Field
from doubtdesk.answering.schemas import DocumentSource


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


class ConversationSummary(BaseModel):
    conversation_id: UUID
    title: str
    created_at: str
    last_accessed_ns: int
    message_count: int


class ConversationMessage(BaseModel):
    id: int
    question: str
    answer: str
    formatted_answer: str
    sources: list[DocumentSource] = Field(default_factory=list)
    doc_ids: list[str]
    status: str
    created_at: str


class ConversationDetail(BaseModel):
    conversation_id: UUID
    title: str
    created_at: str
    last_accessed_ns: int
    messages: list[ConversationMessage]

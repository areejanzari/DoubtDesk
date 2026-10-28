from uuid import UUID
from typing import Literal
from pydantic import BaseModel,Field, field_validator
from doubtdesk.answering.schemas import DocumentSource

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: list[str] = Field(default_factory=list)
    conversation_id: UUID | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, question: str) -> str:
        question = question.strip()

        if not question:
            raise ValueError("Question cannot be empty.")

        if len(question.split()) > 150:
            raise ValueError(
                "Question cannot exceed 150 words."
            )

        return question

class AskResponse(BaseModel):
    status: Literal["grounded", "general_knowledge","out_of_scope"]
    answer: str
    sources: list[DocumentSource]
    formatted_answer: str
    conversation_id: UUID | None = None

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    page_count: int


class UploadResponse(BaseModel):
    doc_id: str
    status: Literal["completed", "cached"]
    page_count: int
    filename: str

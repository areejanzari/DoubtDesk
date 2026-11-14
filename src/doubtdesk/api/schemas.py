from uuid import UUID
from typing import Literal
from pydantic import BaseModel,Field, field_validator
from doubtdesk.answering.schemas import DocumentSource

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    doc_ids: list[str] = Field(default_factory=list)
    conversation_id: UUID | None = None

    @field_validator("doc_ids")
    @classmethod
    def validate_doc_ids(cls, values):
        try:
            unique = list(dict.fromkeys(str(UUID(value)) for value in values))
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError("Document IDs must be valid UUIDs.") from exc
        if len(unique) > 10:
            raise ValueError("At most 10 documents can be selected.")
        return unique

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
    status: Literal["grounded", "general_knowledge", "out_of_scope", "verification_failed"]
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

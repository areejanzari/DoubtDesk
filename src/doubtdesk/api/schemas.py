from typing import Literal
from pydantic import BaseModel,Field
from doubtdesk.answering.schemas import DocumentSource

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    doc_ids: list[str] = Field(default_factory=list)

class AskResponse(BaseModel):
    status: Literal["grounded", "general_knowledge"]
    answer: str
    sources: list[DocumentSource]
    formatted_answer: str
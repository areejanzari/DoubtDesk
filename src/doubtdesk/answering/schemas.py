from typing import Literal
from pydantic import BaseModel, Field

class DocumentSource(BaseModel):
    doc_id: str
    page_numbers: list[int]


class EvidenceResult(BaseModel):
    status: Literal["sufficient", "insufficient"]
    answer: str
    sources: list[DocumentSource] = Field(default_factory=list)
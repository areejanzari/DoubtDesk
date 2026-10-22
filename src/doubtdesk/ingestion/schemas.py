from typing import Literal
from pydantic import BaseModel, Field


class PageRecord(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    text: str
    classification: Literal[
        "good_text",
        "low_text",
        "ocr_candidate",
    ]
    image_coverage: float = Field(ge=0.0, le=1.0)
    image_path: str


class ChunkRecord(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str

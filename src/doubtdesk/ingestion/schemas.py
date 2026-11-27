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
    ocr_status: Literal["not_needed", "ok", "empty", "failed"] = "not_needed"
    figure_area: float = Field(default=0.0, ge=0.0, le=1.0)
    figure_refs: list[str] = Field(default_factory=list)
    inline_image_count: int = Field(default=0, ge=0)
    has_table: bool = False
    watermark_detected: bool = False
    visual: bool = False


class ChunkRecord(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    text: str

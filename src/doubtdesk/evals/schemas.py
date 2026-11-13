from typing import Literal
from pydantic import BaseModel, Field, field_validator


class GoldenQuestion(BaseModel):
    question: str = Field(min_length=3)
    reference_answer: str = Field(min_length=3)
    must_include: list[str] = Field(min_length=1)


class MetricThresholds(BaseModel):
    faithfulness: float = Field(default=0.80, ge=0, le=1)
    answer_relevancy: float = Field(default=0.70, ge=0, le=1)
    context_precision: float = Field(default=0.70, ge=0, le=1)
    context_recall: float = Field(default=0.80, ge=0, le=1)
    answer_correctness: float = Field(default=0.80, ge=0, le=1)


class EvalsRequest(BaseModel):
    doc_ids: list[str] = Field(min_length=1)
    questions: list[GoldenQuestion] = Field(min_length=5, max_length=10)
    mode: Literal["retrieval", "full"] = "full"
    thresholds: MetricThresholds = Field(default_factory=MetricThresholds)

    @field_validator("doc_ids")
    @classmethod
    def validate_doc_ids(cls, value: list[str]):
        if len(value) != len(set(value)):
            raise ValueError("Document IDs must be unique")
        if any(not doc_id.strip() for doc_id in value):
            raise ValueError("Document IDs cannot be empty")
        return value

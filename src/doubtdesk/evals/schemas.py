from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator

FULL_EVAL_MAX_QUESTIONS = 10
RETRIEVAL_EVAL_MAX_QUESTIONS = 100


class ExpectedPage(BaseModel):
    doc_id: str | None = None
    filename: str | None = None
    page_number: int = Field(ge=1)

    @model_validator(mode="after")
    def require_document(self):
        if not (self.doc_id or self.filename):
            raise ValueError("Expected page needs doc_id or filename")
        return self


class GoldenQuestion(BaseModel):
    question: str = Field(min_length=3)
    reference_answer: str = Field(min_length=3)
    must_include: list[str] = Field(min_length=1)
    expected_pages: list[ExpectedPage] = Field(default_factory=list)
    category: str | None = None


class MetricThresholds(BaseModel):
    faithfulness: float = Field(default=0.80, ge=0, le=1)
    answer_relevancy: float = Field(default=0.70, ge=0, le=1)
    context_precision: float = Field(default=0.70, ge=0, le=1)
    context_recall: float = Field(default=0.80, ge=0, le=1)
    answer_correctness: float = Field(default=0.80, ge=0, le=1)


class EvalsRequest(BaseModel):
    doc_ids: list[str] = Field(min_length=1)
    questions: list[GoldenQuestion] = Field(min_length=5, max_length=RETRIEVAL_EVAL_MAX_QUESTIONS)
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

    @model_validator(mode="after")
    def validate_question_count(self):
        if self.mode == "full" and len(self.questions) > FULL_EVAL_MAX_QUESTIONS:
            raise ValueError(f"Full evaluations allow at most {FULL_EVAL_MAX_QUESTIONS} questions")
        return self

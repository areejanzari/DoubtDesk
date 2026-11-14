from typing import TypedDict
from doubtdesk.answering.schemas import EvidenceResult


class DoubtDeskState(TypedDict, total=False):
    user_id: str
    question: str
    doc_ids: list[str]
    retrieved_pages: list[dict]
    retrieved_chunks: list[dict]
    retrieval_candidate_count: int
    evidence_pages: list[dict]
    evidence_mode: str
    evidence_assessment: object
    candidate_error: str
    evidence: EvidenceResult
    status: str
    answer: str
    sources: list[dict]
    formatted_answer: str
    paraphrases: list[str]
    scope_predictions: list[dict]
    is_off_topic: bool
    conversation_id: str
    resolved_question: str
    request_id: str
    
    evidence_tier: str
    document_insufficient: bool
    skip_output_judge: bool
    judge_passed: bool
    judge_attempts: int

    evidence_average: float | None
    judge_reviews: list[dict]
    judge_failure_reason: str | None
    judge_regenerations: int
    judge_needs_fallback: bool
    judge_fallback_attempted: bool
    verification_failed: bool

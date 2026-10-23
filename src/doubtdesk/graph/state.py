from typing import TypedDict
from doubtdesk.answering.schemas import EvidenceResult


class DoubtDeskState(TypedDict, total=False):
    question: str
    doc_ids: list[str]
    retrieved_pages: list[dict]
    evidence: EvidenceResult
    status: str
    answer: str
    sources: list[dict]
    formatted_answer: str

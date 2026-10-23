from google import genai
from doubtdesk.answering.schemas import EvidenceResult
from doubtdesk.answering.fallback import generate_general_answer
from doubtdesk.answering.validation import (
    validate_evidence_consistency,
    validate_evidence_sources,
)
from doubtdesk.answering.vision import (
    build_vision_content,
    generate_vision_answer,
)

def resolve_answer(
        result: EvidenceResult,
        pages: list[dict],
        question: str,
        client: genai.Client,
        model_name: str,
) -> dict:
    validate_evidence_consistency(result)
    validate_evidence_sources(result, pages)

    if result.status == "sufficient":
        return {
            "status": "grounded",
            "answer": result.answer,
            "sources": [
                source.model_dump()
                for source in result.sources
            ]
        }

    general_answer = generate_general_answer(
        client=client,
        model_name=model_name,
        question=question,
    )

    return {
        "status": "general_knowledge",
        "answer": general_answer,
        "sources": [],
    }

def answer_question(
    question: str,
    pages: list[dict],
    client: genai.Client,
    model_name: str,
) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not pages:
        general_answer = generate_general_answer(
            client=client,
            model_name=model_name,
            question=question,
        )

        return {
            "status": "general_knowledge",
            "answer": general_answer,
            "sources": [],
        }

    content = build_vision_content(question, pages)

    evidence = generate_vision_answer(
        client=client,
        model_name=model_name,
        content=content,
    )

    return resolve_answer(
        result=evidence,
        pages=pages,
        question=question,
        client=client,
        model_name=model_name,
    )

from pydantic import BaseModel, Field, field_validator
from google.genai import types
import json
from time import perf_counter
from doubtdesk.analytics.tracker import record_gemini_usage, record_gemini_failure

class ParaphraseResult(BaseModel):
    questions: list[str] = Field(min_length=3,max_length=3)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls,questions: list[str]) -> list[str]:
        if any(not question.strip() for question in questions):
            raise ValueError("Paraphrases cannot be empty.")
        return questions

class ContextualParaphraseResult(ParaphraseResult):
    resolved_question: str = Field(min_length=1)


def resolve_and_paraphrase(
    client,
    model_name: str,
    question: str,
    history: list[dict],
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
) -> ContextualParaphraseResult:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    prompt = """
    You are the conversational question resolver for DoubtDesk.

    Given the current question and previous conversation context:

    1. Determine whether the current question refers to an earlier topic.
    2. If it does, rewrite it as a standalone question.
    3. If it is independent, preserve its original intent.
    4. Generate exactly three distinct paraphrases of the resolved question.
    5. Correct spelling and grammatical errors.
    6. The third paraphrase must focus on the core concept.
    7. Never add assumptions or change the user's requested task.
    8. Do not transform non-academic questions into academic ones.
    9. If a reference is ambiguous, do not invent missing context.
    10. Do not answer the question.
    """

    content = json.dumps({
        "history": history,
        "current_question": question,
    })

    start = perf_counter()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=content,
            config=types.GenerateContentConfig(
                system_instruction=prompt,
                response_mime_type="application/json",
                response_schema=ContextualParaphraseResult,
                temperature=0.3,
            ),
        )
            
    except Exception as exc:
        latency_ms = round((perf_counter() - start) * 1000)

        if request_id is not None:
            record_gemini_failure(
                request_id=request_id,
                conversation_id=conversation_id,
                operation="paraphrasing",
                model=model_name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )

        raise

    latency_ms = round((perf_counter() - start) * 1000)

    try:
        if response.parsed is None:
            raise ValueError("Gemini did not return valid paraphrases.")

        result = ContextualParaphraseResult.model_validate(
            response.parsed
        )

    except ValueError as exc:
        if request_id is not None:
            record_gemini_failure(
                request_id=request_id,
                conversation_id=conversation_id,
                operation="paraphrasing",
                model=model_name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
                response=response,
            )

        raise

    if request_id is not None:
        record_gemini_usage(
            response=response,
            request_id=request_id,
            conversation_id=conversation_id,
            operation="paraphrasing",
            model=model_name,
            latency_ms=latency_ms,
        )

    return result
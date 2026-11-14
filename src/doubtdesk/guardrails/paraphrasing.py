import json
from pydantic import BaseModel, Field, field_validator
from doubtdesk.observability.tracing import observe_llm_response, trace_generation


class ParaphraseResult(BaseModel):
    questions: list[str] = Field(min_length=3, max_length=3)

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, questions: list[str]) -> list[str]:
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

    instruction = (
        "You resolve conversational student questions for DoubtDesk. "
        "Use prior conversation context only to resolve explicit references. "
        "Rewrite the question so it stands alone, preserving intent. "
        "Return exactly three distinct paraphrases. "
        "Correct spelling without changing meaning; the third paraphrase focuses on the core concept. "
        "Do not turn an off-topic question into an academic question. "
        "Do not invent missing information or answer the question."
    )

    with trace_generation("paraphrasing", model_name, request_id, input={"question": question, "history_count": len(history)}):
        response = client.responses.parse(
            model=model_name,
            input=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps({"history": history, "current_question": question})},
            ],
            text_format=ContextualParaphraseResult,
            reasoning={"effort": "none"},
            max_output_tokens=650,
        )
        observe_llm_response("paraphrasing", response, model_name, request_id, conversation_id)
    if response.output_parsed is None:
        raise ValueError("Model returned no valid paraphrases")
    return ContextualParaphraseResult.model_validate(response.output_parsed)

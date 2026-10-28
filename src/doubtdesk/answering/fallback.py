from google import genai
from google.genai import types
from time import perf_counter
from doubtdesk.analytics.tracker import record_gemini_usage, record_gemini_failure

def generate_general_answer(
    client: genai.Client,
    model_name: str,
    question: str,
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    start = perf_counter()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are DoubtDesk, a STEM tutor. "
                    "Answer the student's question accurately and "
                    "clearly using general knowledge. "
                    "Keep your answer within approximately 300 words. "
                    "Be concise but provide enough explanation for students "
                    "to understand the underlying concepts. "
                    "Preserve essential formulas, equations, and examples. "
                    "Do not claim that the answer comes from "
                    "an uploaded document. "
                    "Do not invent document citations."
                ),
                temperature=0,
                max_output_tokens=1024,
            ),
        )

    except Exception as exc:
        latency_ms = round((perf_counter() - start) * 1000)

        if request_id is not None:
            record_gemini_failure(
                request_id=request_id,
                conversation_id=conversation_id,
                operation="general_answer",
                model=model_name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )

        raise

    latency_ms = round((perf_counter() - start) * 1000)

    try:
        answer = response.text

        if not answer:
            raise RuntimeError(
                "The AI model could not generate an answer."
            )

    except (RuntimeError, ValueError) as exc:
        if request_id is not None:
            record_gemini_failure(
                request_id=request_id,
                conversation_id=conversation_id,
                operation="general_answer",
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
            operation="general_answer",
            model=model_name,
            latency_ms=latency_ms,
        )

    return answer

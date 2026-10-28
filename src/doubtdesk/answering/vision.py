from pathlib import Path
from time import perf_counter
from google import genai
from google.genai import types
from doubtdesk.answering.schemas import EvidenceResult
from doubtdesk.analytics.tracker import record_gemini_usage, record_gemini_failure


def load_page_image(image_path: Path) -> types.Part:
    if not image_path.is_file():
        raise FileNotFoundError(
            f"Page image not found: {image_path}"
        )

    image_bytes = image_path.read_bytes()

    return types.Part.from_bytes(
        data=image_bytes,
        mime_type="image/png",
    )


def build_vision_content(
    question: str,
    pages: list[dict],
) -> list:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    if not pages:
        raise ValueError("At least one page is required")

    content = [
        (
            "You are DoubtDesk, a STEM tutor.\n\n"
            "Your task is to determine whether the attached "
            "PDF pages contain sufficient evidence to answer "
            "the student's question.\n\n"
            "Rules:\n"
            "1. Use ONLY information supported by the "
            "attached PDF pages.\n"
            "2. If the pages provide sufficient evidence, "
            "set status='sufficient', write the answer, "
            "and cite the supporting document IDs and "
            "page numbers.\n"
            "3. If the pages do not provide sufficient evidence, "
            "set status='insufficient', answer='', "
            "and sources=[].\n"
            "4. Do not use outside knowledge to fill gaps "
            "in the document evidence.\n"
            "5. Cite ONLY document IDs and page numbers "
            "provided with the images.\n"
            "6. Do not invent facts or citations.\n"
            "7. Explain STEM concepts clearly when the "
            "document evidence supports them.\n"
            "8. Keep the answer within approximately 300 words. "
            "Prioritize relevant explanations, formulas, and essential "
            "details from the document.\n"
            "9. Preserve accurate document citations and the "
            "required structured response format.\n\n"
            f"Student question: {question}"
        )
    ]

    for page in pages:
        content.append(
            f"Document ID: {page['doc_id']}, "
            f"PDF page number: {page['page_number']}"
        )
        content.append(load_page_image(page["image_path"]))

    return content


def generate_vision_answer(
    client: genai.Client,
    model_name: str,
    content: list,
    request_id: str | None = None,
    conversation_id: str | None = None,
) -> EvidenceResult:
    if not content:
        raise ValueError("Content cannot be empty.")

    start = perf_counter()

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=content,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=EvidenceResult,
                temperature=0,
                max_output_tokens=1536,
            ),
        )

    except Exception as exc:
        latency_ms = round((perf_counter() - start) * 1000)

        if request_id is not None:
            record_gemini_failure(
                request_id=request_id,
                conversation_id=conversation_id,
                operation="vision",
                model=model_name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )

        raise

    latency_ms = round((perf_counter() - start) * 1000)

    try:
        if response.parsed is None:
            raise RuntimeError(
                "The AI model did not return a valid structured response."
            )

        result = EvidenceResult.model_validate(response.parsed)

    except (ValueError, RuntimeError) as exc:
        if request_id is not None:
            record_gemini_failure(
                request_id=request_id,
                conversation_id=conversation_id,
                operation="vision",
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
            operation="vision",
            model=model_name,
            latency_ms=latency_ms,
        )

    return result


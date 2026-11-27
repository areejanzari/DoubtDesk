import asyncio
from doubtdesk.llm import acall
from doubtdesk.observability.tracing import observe_llm_response, trace_generation


ANSWER_MAX_OUTPUT_TOKENS = 800


GENERAL_INSTRUCTIONS = (
    "You are DoubtDesk, an academic tutor. Answer accurately and clearly using general knowledge "
    "in about 250 output tokens, or up to 600 for derivations or multi-step numerical problems, showing each step. "
    "Preserve essential formulas and examples. "
    "Do not claim to cite an uploaded PDF or fabricate PDF page references."
)


def general_input(question: str, feedback: str | None = None) -> list[dict]:
    instruction = GENERAL_INSTRUCTIONS
    if feedback:
        instruction += " Revise your previous answer using this reviewer feedback: " + feedback[:1200]
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": question},
    ]


def _general_request(model_name: str, question: str, feedback: str | None) -> dict:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    return {
        "model": model_name,
        "input": general_input(question, feedback),
        "reasoning": {"effort": "none"},
        "max_output_tokens": ANSWER_MAX_OUTPUT_TOKENS,
    }


def _general_result(response) -> str:
    answer = response.output_text
    if not answer:
        raise RuntimeError("Model returned no answer")
    return answer


def generate_general_answer(
    client,
    model_name: str,
    question: str,
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    feedback: str | None = None,
) -> str:
    request = _general_request(model_name, question, feedback)
    with trace_generation("general_answer", model_name, request_id, input={"question": question, "regeneration": bool(feedback)}):
        response = client.responses.create(**request)
        observe_llm_response("general_answer", response, model_name, request_id, conversation_id)
    return _general_result(response)


async def agenerate_general_answer(
    client,
    model_name: str,
    question: str,
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    feedback: str | None = None,
) -> str:
    request = _general_request(model_name, question, feedback)
    with trace_generation("general_answer", model_name, request_id, input={"question": question, "regeneration": bool(feedback)}):
        response = await acall(client.responses.create, **request)
        await asyncio.to_thread(observe_llm_response, "general_answer", response, model_name, request_id, conversation_id)
    return _general_result(response)

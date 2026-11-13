from doubtdesk.observability.tracing import observe_llm_response


GENERAL_INSTRUCTIONS = (
    "You are DoubtDesk, an academic tutor. Answer accurately and clearly using general knowledge "
    "in 300 output tokens or fewer. Preserve essential formulas and examples. "
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


def generate_general_answer(
    client,
    model_name: str,
    question: str,
    *,
    request_id: str | None = None,
    conversation_id: str | None = None,
    feedback: str | None = None,
) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    response = client.responses.create(
        model=model_name,
        input=general_input(question, feedback),
        reasoning={"effort": "none"},
        max_output_tokens=300,
    )
    observe_llm_response("general_answer", response, model_name, request_id, conversation_id)
    answer = response.output_text
    if not answer:
        raise RuntimeError("Model returned no answer")
    return answer

from google import genai
from google.genai import types

def generate_general_answer(
    client: genai.Client,
    model_name: str,
    question: str,
) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty")

    response = client.models.generate_content(
        model=model_name,
        contents=question,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are DoubtDesk, a STEM tutor. "
                "Answer the student's question accurately and "
                "clearly using general knowledge. "
                "Do not claim that the answer comes from "
                "an uploaded document. "
                "Do not invent document citations."
            ),
            temperature=0,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "The AI model could not generate an answer."
        )

    return response.text

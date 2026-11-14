import json
import re
from types import SimpleNamespace

from doubtdesk.answering.evidence_selection import TextEvidenceAssessment
from doubtdesk.answering.output_guardrails import JudgeVerdict
from doubtdesk.answering.schemas import ModelEvidenceResult
from doubtdesk.guardrails.paraphrasing import ContextualParaphraseResult


class FakeStream:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="response.output_text.delta", delta=self.text)

    def get_final_response(self):
        return SimpleNamespace(usage=None)


class FakeResponses:
    @staticmethod
    def _answer(input_messages):
        content = json.dumps(input_messages)
        match = re.search(r"Source alias: (S\d+), PDF page number: (\d+)", content)
        if match:
            return {
                "status": "sufficient",
                "answer": "Photosynthesis converts light energy into chemical energy.",
                "sources": [match.group(1)],
            }
        return {"status": "insufficient", "answer": "", "sources": []}

    def parse(self, *, model, input, text_format, **kwargs):
        if text_format is ContextualParaphraseResult:
            question = json.loads(input[1]["content"])["current_question"]
            result = {
                "resolved_question": question,
                "questions": [question, f"Explain {question}", f"What does {question} mean?"],
            }
        elif text_format is TextEvidenceAssessment:
            payload = json.loads(input[1]["content"])
            score = 0.95 if payload["retrieved_text"] else 0.05
            result = {
                "sufficiency_score": score,
                "recommended_image_pages": [],
            }
        elif text_format is ModelEvidenceResult:
            result = self._answer(input)
        else:
            result = None
        return SimpleNamespace(output_parsed=result, output_text="E2E response.", usage=None)

    def stream(self, *, model, input, text_format=None, **kwargs):
        if text_format is ModelEvidenceResult:
            return FakeStream(json.dumps(self._answer(input)))
        return FakeStream("This is a general knowledge answer.")


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponses()

    def close(self):
        return None


class FakeOutputJudge:
    def evaluate(self, *, question, result, chunks=None, pages=None):
        return JudgeVerdict(
            passed=True, relevance=1, groundedness=1, completeness=1,
            safety="pass", feedback="E2E fixture approved.",
        )

import json
import re
from types import SimpleNamespace
from doubtdesk.answering.schemas import EvidenceResult
from doubtdesk.guardrails.paraphrasing import ContextualParaphraseResult


class FakeModels:
    def generate_content(self, *, model, contents, config):
        response_schema = getattr(config, "response_schema", None)

        if response_schema is ContextualParaphraseResult:
            payload = json.loads(contents)
            question = payload["current_question"]

            return self._response(
                parsed={
                    "resolved_question": question,
                    "questions": [
                        question,
                        f"Explain {question}",
                        f"What is the core concept behind {question}",
                    ],
                }
            )

        if response_schema is EvidenceResult:
            text = "\n".join(
                item for item in contents if isinstance(item, str)
            )

            matches = re.findall(
                r"Document ID: ([\w-]+), PDF page number: (\d+)",
                text,
            )

            if matches:
                doc_id, page_number = matches[0]

                return self._response(
                    parsed={
                        "status": "sufficient",
                        "answer": "Photosynthesis converts light energy into chemical energy.",
                        "sources": [
                            {
                                "doc_id": doc_id,
                                "page_numbers": [int(page_number)],
                            }
                        ],
                    }
                )

            return self._response(
                parsed={
                    "status": "insufficient",
                    "answer": "",
                    "sources": [],
                }
            )

        return self._response(
            text="This is a general knowledge answer."
        )

    @staticmethod
    def _response(parsed=None, text=None):
        return SimpleNamespace(
            parsed=parsed,
            text=text,
            usage_metadata=SimpleNamespace(
                prompt_token_count=0,
                candidates_token_count=0,
                thoughts_token_count=0,
                cached_content_token_count=0,
                total_token_count=0,
            ),
        )


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeModels()

    def close(self):
        pass

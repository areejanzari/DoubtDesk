import re
from types import SimpleNamespace


class FakeModels:
    def generate_content(self, *, model, contents, config):
        text = "\n".join(
            item for item in contents if isinstance(item, str)
        )

        matches = re.findall(
            r"Document ID: ([\w-]+), PDF page number: (\d+)",
            text,
        )

        if matches:
            doc_id, page_number = matches[0]

            return SimpleNamespace(
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

        return SimpleNamespace(
            text="This is a general knowledge answer."
        )


class FakeGeminiClient:
    def __init__(self):
        self.models = FakeModels()

    def close(self):
        pass
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


MODEL_NAME = "govtech/stsb-roberta-base-off-topic"
TOKENIZER_NAME = "cross-encoder/stsb-roberta-base"
MODEL_FILE = "models/off-topic-cross-encoder-stsb-roberta-base-CrossEncoder.onnx"

SYSTEM = "DoubtDesk is a tutor that answers academic questions in biology, chemistry, physics, and mathematics only."


class ScopeClassifier:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        model_path = hf_hub_download(
            repo_id=MODEL_NAME,
            filename=MODEL_FILE,
        )

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

    def predict(self, questions: list[str]) -> list[dict]:
        predictions = []

        for question in questions:
            encoded = self.tokenizer(
                SYSTEM,
                question,
                return_tensors="np",
                truncation=True,
                padding="max_length",
                max_length=512,
                return_token_type_ids=False,
            )

            model_inputs = self.session.get_inputs()

            inputs = {
                model_inputs[0].name: encoded["input_ids"].astype(np.int64),
                model_inputs[1].name: encoded["attention_mask"].astype(np.int64),
            }

            logits = self.session.run(None, inputs)[0][0]
            shifted = logits - np.max(logits)
            probabilities = np.exp(shifted) / np.exp(shifted).sum()

            predictions.append({
                "question": question,
                "label": int(np.argmax(probabilities)),
                "off_topic_probability": round(float(probabilities[1]), 4),
            })

        return predictions


def load_scope_classifier() -> ScopeClassifier:
    return ScopeClassifier()

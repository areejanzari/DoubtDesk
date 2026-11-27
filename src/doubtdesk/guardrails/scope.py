import logging
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from doubtdesk.config import settings
from doubtdesk.runtime import inference_slot, ort_session_options


logger = logging.getLogger(__name__)

MODEL_NAME = "govtech/stsb-roberta-base-off-topic"
TOKENIZER_NAME = "cross-encoder/stsb-roberta-base"
MODEL_FILE = "models/off-topic-cross-encoder-stsb-roberta-base-CrossEncoder.onnx"
FIXED_MASK_CONSTANTS = {
    "/base_model/Constant_13_output_0": 1,
    "/base_model/Constant_16_output_0": 512,
}

SYSTEM = "DoubtDesk is a tutor that answers academic questions in biology, chemistry, physics, and mathematics only."


def _dynamic_model_path(model_path: str) -> Path | None:
    source = Path(model_path)
    stat = source.stat()
    target = settings.model_cache_dir / f"scope-dynamic-{stat.st_size}-{int(stat.st_mtime)}.onnx"
    if target.exists():
        return target
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(source))
    patched = {}
    for node in model.graph.node:
        expected = FIXED_MASK_CONSTANTS.get(node.output[0]) if node.op_type == "Constant" else None
        if expected is None:
            continue
        tensor = node.attribute[0].t
        value = numpy_helper.to_array(tensor)
        if value.size != 1 or int(value.ravel()[0]) != expected:
            return None
        tensor.CopyFrom(numpy_helper.from_array(np.ones_like(value), tensor.name))
        patched[node.output[0]] = True
    if len(patched) != len(FIXED_MASK_CONSTANTS):
        return None
    for graph_input in model.graph.input[:2]:
        dims = graph_input.type.tensor_type.shape.dim
        dims[0].ClearField("dim_value")
        dims[0].dim_param = "batch_size"
        dims[1].ClearField("dim_value")
        dims[1].dim_param = "sequence_length"
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(f".{os.getpid()}.tmp")
    onnx.save(model, str(partial))
    del model
    os.replace(partial, target)
    return target


def _load_session(model_path: str) -> ort.InferenceSession:
    try:
        dynamic_path = _dynamic_model_path(model_path)
        if dynamic_path is not None:
            return ort.InferenceSession(str(dynamic_path), sess_options=ort_session_options(),
                                        providers=["CPUExecutionProvider"])
        logger.warning("Scope model layout changed; using fixed-length session")
    except Exception:
        logger.exception("Could not prepare dynamic-length scope model; using fixed-length session")
    return ort.InferenceSession(model_path, sess_options=ort_session_options(), providers=["CPUExecutionProvider"])


class PairTokenizer:
    def __init__(self, path: str, max_length: int = 512):
        self.tokenizer = Tokenizer.from_file(path)
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
        self.max_length = max_length
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.pad_id is None:
            raise RuntimeError("Scope tokenizer has no <pad> token")

    def __call__(self, firsts, seconds, *, padding=True, max_length=None, **kwargs):
        if max_length is not None and max_length != self.max_length:
            raise ValueError("Unsupported tokenizer max_length")
        encodings = self.tokenizer.encode_batch(list(zip(firsts, seconds)))
        width = self.max_length if padding == "max_length" else max(len(item.ids) for item in encodings)
        input_ids = np.full((len(encodings), width), self.pad_id, dtype=np.int64)
        attention_mask = np.zeros((len(encodings), width), dtype=np.int64)
        for row, item in enumerate(encodings):
            input_ids[row, :len(item.ids)] = item.ids
            attention_mask[row, :len(item.ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ScopeClassifier:
    def __init__(self):
        self.tokenizer = PairTokenizer(hf_hub_download(repo_id=TOKENIZER_NAME, filename="tokenizer.json"))

        model_path = hf_hub_download(
            repo_id=MODEL_NAME,
            filename=MODEL_FILE,
        )

        self.session = _load_session(model_path)

    def predict(self, questions: list[str]) -> list[dict]:
        if not questions:
            return []
        model_inputs = self.session.get_inputs()
        shapes = [item.shape for item in model_inputs[:2]]
        fixed_batches = {shape[0] for shape in shapes if isinstance(shape[0], int) and shape[0] > 0}
        fixed_lengths = {shape[1] for shape in shapes if isinstance(shape[1], int) and shape[1] > 0}
        if len(fixed_batches) > 1 or len(fixed_lengths) > 1:
            raise RuntimeError("Inconsistent ONNX input shapes")
        fixed_batch = next(iter(fixed_batches), None)
        batch_size = fixed_batch or len(questions)
        fixed_length = next(iter(fixed_lengths), None)
        predictions = []
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start+batch_size]
            padded = batch + [batch[-1]] * (batch_size-len(batch)) if fixed_batch else batch
            encoded = self.tokenizer(
                [SYSTEM] * len(padded), padded, return_tensors="np", truncation=True,
                padding="max_length" if fixed_length else True, max_length=fixed_length or 512,
                return_token_type_ids=False,
            )
            inputs = {
                model_inputs[0].name: encoded["input_ids"].astype(np.int64),
                model_inputs[1].name: encoded["attention_mask"].astype(np.int64),
            }
            with inference_slot():
                logits_batch = self.session.run(None, inputs)[0]
            for question, logits in zip(batch, logits_batch):
                shifted = logits - np.max(logits)
                probabilities = np.exp(shifted) / np.exp(shifted).sum()
                predictions.append({"question": question, "label": int(np.argmax(probabilities)),
                                    "off_topic_probability": round(float(probabilities[1]), 4)})
        return predictions


def load_scope_classifier() -> ScopeClassifier:
    return ScopeClassifier()

import numpy as np
from doubtdesk.config import runtime_settings
from doubtdesk.runtime import inference_slot, ort_session_options

RERANK_MAX_TOKENS = 512
RERANK_BATCH = 4


class OnnxCrossEncoder:
    def __init__(self, repo: str | None = None):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        repo = repo or runtime_settings().reranker_model
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=RERANK_MAX_TOKENS, strategy="only_second")
        self.pad_id = self.tokenizer.token_to_id("[PAD]") or 0
        self.session = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"), sess_options=ort_session_options(),
                                            providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}

    def _run(self, encodings) -> np.ndarray:
        width = max(len(item.ids) for item in encodings)
        ids = np.full((len(encodings), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        types = np.zeros((len(encodings), width), dtype=np.int64)
        for row, item in enumerate(encodings):
            ids[row, :len(item.ids)] = item.ids
            mask[row, :len(item.ids)] = 1
            types[row, :len(item.ids)] = item.type_ids
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        logits = self.session.run(None, {name: value for name, value in feeds.items() if name in self.input_names})[0]
        return np.asarray(logits, dtype=np.float32).reshape(len(encodings), -1)[:, 0]

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        encodings = self.tokenizer.encode_batch([(query, passage) for passage in passages])
        order = sorted(range(len(encodings)), key=lambda index: len(encodings[index].ids))
        scores = [0.0] * len(encodings)
        for start in range(0, len(order), RERANK_BATCH):
            group = order[start:start + RERANK_BATCH]
            with inference_slot():
                values = self._run([encodings[index] for index in group])
            for index, value in zip(group, values):
                scores[index] = float(value)
        return scores


def load_reranker():
    return OnnxCrossEncoder()


def rerank_chunks(reranker, query: str, chunks: list[dict]) -> list[dict]:
    if reranker is None or not chunks:
        return chunks
    scores = reranker.score(query, [chunk["text"] for chunk in chunks])
    ranked = [{**chunk, "rerank_score": score} for chunk, score in zip(chunks, scores)]
    return sorted(ranked, key=lambda item: (-item["rerank_score"], -item.get("rrf_score", 0.0), item["chunk_id"]))

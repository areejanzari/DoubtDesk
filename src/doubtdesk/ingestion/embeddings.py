import json

import numpy as np
from doubtdesk.ingestion.schemas import ChunkRecord
from doubtdesk.runtime import encode, ort_session_options

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_REPO = f"sentence-transformers/{EMBEDDING_MODEL_NAME}"


class OnnxEmbeddingModel:
    def __init__(self, repo: str = EMBEDDING_REPO):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        with open(hf_hub_download(repo, "sentence_bert_config.json"), encoding="utf-8") as handle:
            self.max_seq_length = json.load(handle)["max_seq_length"]
        self.tokenizer = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        self.tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=self.max_seq_length)
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        self.session = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"), sess_options=ort_session_options(),
                                            providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.dimension = self.session.get_outputs()[0].shape[-1]

    def get_sentence_embedding_dimension(self) -> int:
        return self.dimension

    def _batch(self, encodings) -> np.ndarray:
        width = max(len(item.ids) for item in encodings)
        ids = np.full((len(encodings), width), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(encodings), width), dtype=np.int64)
        types = np.zeros((len(encodings), width), dtype=np.int64)
        for row, item in enumerate(encodings):
            ids[row, :len(item.ids)] = item.ids
            mask[row, :len(item.ids)] = 1
            types[row, :len(item.ids)] = item.type_ids
        feeds = {"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
        hidden = self.session.run(None, {name: value for name, value in feeds.items() if name in self.input_names})[0]
        weights = mask[..., None].astype(hidden.dtype)
        return (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)

    def encode(self, sentences, normalize_embeddings: bool = False, batch_size: int = 32, **kwargs) -> np.ndarray:
        single = isinstance(sentences, str)
        texts = [sentences] if single else list(sentences)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        encodings = self.tokenizer.encode_batch(texts)
        order = np.argsort([-len(item.ids) for item in encodings], kind="stable")
        output = np.empty((len(texts), self.dimension), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            selected = order[start:start + batch_size]
            output[selected] = self._batch([encodings[index] for index in selected])
        if normalize_embeddings:
            output /= np.clip(np.linalg.norm(output, axis=1, keepdims=True), 1e-12, None)
        return output[0] if single else output


def load_embedding_model():
    return OnnxEmbeddingModel()


_FALLBACK_TOKENIZER = None


def embedding_tokenizer(model=None):
    global _FALLBACK_TOKENIZER
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "to_str"):
        return tokenizer
    if _FALLBACK_TOKENIZER is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        _FALLBACK_TOKENIZER = Tokenizer.from_file(hf_hub_download(EMBEDDING_REPO, "tokenizer.json"))
    return _FALLBACK_TOKENIZER

def embed_chunks(
        chunks: list[ChunkRecord],
        model,
) -> np.ndarray:
    if not chunks:
        return np.empty((0, model.get_sentence_embedding_dimension()))

    texts = [chunk.text for chunk in chunks]

    embeddings = encode(model, texts)

    return embeddings

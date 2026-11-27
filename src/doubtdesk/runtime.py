import os
import threading
from contextlib import contextmanager

from doubtdesk.config import settings

ENCODE_BATCH = 64

_gate = threading.BoundedSemaphore(settings.model_concurrency)
_configured = False


def configure_threads() -> None:
    global _configured
    if _configured:
        return
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.model_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(settings.model_threads))
    _configured = True


def ort_session_options():
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = settings.model_threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = False
    return options


@contextmanager
def inference_slot():
    with _gate:
        yield


def encode(model, texts: list[str], **kwargs):
    if len(texts) <= ENCODE_BATCH:
        with inference_slot():
            return model.encode(texts, normalize_embeddings=True, **kwargs)
    import numpy as np
    parts = []
    for start in range(0, len(texts), ENCODE_BATCH):
        with inference_slot():
            parts.append(model.encode(texts[start:start + ENCODE_BATCH], normalize_embeddings=True, **kwargs))
    return np.concatenate(parts)

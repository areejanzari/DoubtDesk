import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from doubtdesk import runtime
from doubtdesk.guardrails import scope


class CountingModel:
    def __init__(self):
        self.batches = []
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def encode(self, texts, normalize_embeddings=False):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        with self.lock:
            self.active -= 1
        self.batches.append(len(texts))
        return np.array([[float(len(text))] for text in texts])


def test_encode_splits_large_inputs_and_preserves_order():
    model = CountingModel()
    texts = ["x" * (index % 7 + 1) for index in range(150)]
    result = runtime.encode(model, texts)
    assert model.batches == [64, 64, 22]
    assert result[:, 0].tolist() == [float(len(text)) for text in texts]


def test_encode_limits_concurrent_inference():
    model = CountingModel()
    threads = [threading.Thread(target=runtime.encode, args=(model, ["a"])) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert model.peak <= runtime.settings.model_concurrency


def fake_scope_model(path, mask_value=512):
    nodes = [
        helper.make_node("Constant", [], ["/base_model/Constant_13_output_0"], name="/base_model/Constant_13",
                         value=numpy_helper.from_array(np.array([1], dtype=np.int64))),
        helper.make_node("Constant", [], ["/base_model/Constant_16_output_0"], name="/base_model/Constant_16",
                         value=numpy_helper.from_array(np.array([mask_value], dtype=np.int64))),
        helper.make_node("Identity", ["input"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes, "scope",
        [helper.make_tensor_value_info("input", TensorProto.INT64, [None, 512]),
         helper.make_tensor_value_info("attention_mask", TensorProto.INT64, [1, 512])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [None, 512])],
    )
    onnx.save(helper.make_model(graph), str(path))


def test_scope_model_is_patched_to_dynamic_length_and_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(scope.settings, "model_cache_dir", tmp_path / "cache")
    source = tmp_path / "scope.onnx"
    fake_scope_model(source)
    patched = scope._dynamic_model_path(str(source))
    model = onnx.load(str(patched))
    dims = [[dim.dim_param or dim.dim_value for dim in item.type.tensor_type.shape.dim] for item in model.graph.input]
    assert dims == [["batch_size", "sequence_length"], ["batch_size", "sequence_length"]]
    values = {node.output[0]: int(numpy_helper.to_array(node.attribute[0].t)[0]) for node in model.graph.node if node.op_type == "Constant"}
    assert set(values.values()) == {1}
    with patch.object(onnx, "load", side_effect=AssertionError("cache not used")):
        assert scope._dynamic_model_path(str(source)) == patched


def test_scope_model_with_unexpected_layout_is_not_patched(tmp_path, monkeypatch):
    monkeypatch.setattr(scope.settings, "model_cache_dir", tmp_path / "cache")
    source = tmp_path / "scope.onnx"
    fake_scope_model(source, mask_value=256)
    assert scope._dynamic_model_path(str(source)) is None


def test_scope_session_falls_back_to_original_model(tmp_path, monkeypatch):
    created = []
    monkeypatch.setattr(scope, "_dynamic_model_path", lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(scope.ort, "InferenceSession", lambda path, **kwargs: created.append((path, kwargs)) or "session")
    assert scope._load_session("original.onnx") == "session"
    assert created[0][0] == "original.onnx"
    options = created[0][1]["sess_options"]
    assert options.intra_op_num_threads == runtime.settings.model_threads and not options.enable_cpu_mem_arena


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids
        self.type_ids = [0] * len(ids)


def test_pair_tokenizer_pads_longest_or_fixed_length():
    tokenizer = scope.PairTokenizer.__new__(scope.PairTokenizer)
    tokenizer.tokenizer = SimpleNamespace(encode_batch=lambda pairs: [FakeEncoding(list(range(len(b)))) for _, b in pairs])
    tokenizer.max_length = 6
    tokenizer.pad_id = 1
    dynamic = tokenizer(["s", "s"], ["abc", "a"], padding=True, max_length=6)
    assert dynamic["input_ids"].tolist() == [[0, 1, 2], [0, 1, 1]]
    assert dynamic["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    fixed = tokenizer(["s"], ["ab"], padding="max_length", max_length=6)
    assert fixed["input_ids"].shape == (1, 6) and fixed["attention_mask"].sum() == 2
    with pytest.raises(ValueError):
        tokenizer(["s"], ["a"], max_length=128)


def test_onnx_embedder_mean_pools_sorted_batches_back_into_input_order():
    from doubtdesk.ingestion.embeddings import OnnxEmbeddingModel
    model = OnnxEmbeddingModel.__new__(OnnxEmbeddingModel)
    lengths = {"short": 1, "longest text": 4, "mid one": 2}
    model.tokenizer = SimpleNamespace(encode_batch=lambda texts: [FakeEncoding([7] * lengths[text]) for text in texts])
    model.pad_id = 0
    model.dimension = 2
    model.input_names = {"input_ids", "attention_mask", "token_type_ids"}
    seen_widths = []

    def run(_, feeds):
        ids = feeds["input_ids"]
        seen_widths.append(ids.shape[1])
        hidden = np.stack([np.stack([feeds["attention_mask"].sum(axis=1)[row] * np.ones(2)] * ids.shape[1]) for row in range(len(ids))])
        return [hidden.astype(np.float32)]

    model.session = SimpleNamespace(run=run)
    texts = ["short", "longest text", "mid one"]
    raw = model.encode(texts, batch_size=2)
    assert raw[:, 0].tolist() == [1.0, 4.0, 2.0]
    assert seen_widths == [4, 1]
    normalized = model.encode(texts, normalize_embeddings=True)
    assert np.allclose(np.linalg.norm(normalized, axis=1), 1.0)
    assert model.encode("mid one").shape == (2,)
    assert model.encode([]).shape == (0, 2)
    assert model.get_sentence_embedding_dimension() == 2


def test_embedding_loader_uses_onnx_without_fallback(monkeypatch):
    from doubtdesk.ingestion import embeddings
    monkeypatch.setattr(embeddings, "OnnxEmbeddingModel", lambda: "onnx-model")
    assert embeddings.load_embedding_model() == "onnx-model"
    monkeypatch.setattr(embeddings, "OnnxEmbeddingModel", lambda: (_ for _ in ()).throw(OSError("offline")))
    with pytest.raises(OSError):
        embeddings.load_embedding_model()


def test_judge_loads_other_models_with_onnx(monkeypatch):
    from doubtdesk.evals import judge
    repos = []
    monkeypatch.setattr(judge, "OnnxEmbeddingModel", lambda repo: repos.append(repo) or SimpleNamespace(encode=None))
    assert isinstance(judge.judge_embeddings("all-mpnet-base-v2"), judge.SharedHuggingFaceEmbeddings)
    judge.judge_embeddings("BAAI/bge-small-en-v1.5", SimpleNamespace())
    assert repos == ["sentence-transformers/all-mpnet-base-v2", "BAAI/bge-small-en-v1.5"]


def test_app_import_does_not_load_torch():
    import subprocess
    code = ("import sys, doubtdesk.api.main, doubtdesk.evals.judge; "
            "print(sorted(m for m in ('torch', 'sentence_transformers', 'transformers') if m in sys.modules))")
    output = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert output.strip().splitlines()[-1] == "[]"


def test_judge_reuses_app_embedding_model_for_matching_name():
    from doubtdesk.evals.judge import SharedHuggingFaceEmbeddings, judge_embeddings
    model = SimpleNamespace(encode=lambda text, normalize_embeddings=True: np.array([1.0, 0.0]))
    shared = judge_embeddings("sentence-transformers/all-MiniLM-L6-v2", model)
    assert isinstance(shared, SharedHuggingFaceEmbeddings)
    assert shared.model_instance is model
    assert shared.embed_text("hello") == [1.0, 0.0]

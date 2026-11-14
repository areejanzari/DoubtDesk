import json
from types import SimpleNamespace
from doubtdesk.api.streaming import sse, stream_general, stream_answer, stream_grounded


class FakeStream:
    def __init__(self, text, size=5):
        self.text = text
        self.size = size

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        for start in range(0, len(self.text), self.size):
            yield SimpleNamespace(type="response.output_text.delta", delta=self.text[start:start+self.size])

    def get_final_response(self):
        return SimpleNamespace(usage=None)


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.responses = self
        self.calls = 0
        self.inputs = []

    def stream(self, **kwargs):
        self.calls += 1
        self.inputs.append(kwargs)
        return FakeStream(self.text)


def collect_generator(generator):
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as complete:
            return events, complete.value


def test_general_stream_does_not_add_model_call():
    client = FakeClient("An answer")
    events, result = collect_generator(stream_general(client, "gpt-6-luna", "What is gravity?"))
    assert "".join(item[1]["text"] for item in events) == "An answer"
    assert result["answer"] == "An answer"
    assert client.calls == 1


def test_grounded_stream_validates_sources(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"fake png bytes")
    page = {"doc_id": "document", "page_number": 2, "image_path": image}
    raw = json.dumps({"status": "sufficient", "answer": "Valid result", "sources": ["S1"]})
    client = FakeClient(raw)
    result = stream_grounded(client, "gpt-6-luna", "Question?", [page])
    assert result["answer"] == "Valid result"
    assert result["sources"] == [{"doc_id": "document", "page_numbers": [2]}]
    assert client.calls == 1


def test_sse_wire_format():
    assert sse("token", {"text": "Hello"}) == 'event: token\ndata: {"text": "Hello"}\n\n'


def test_text_grounded_stream_does_not_send_images():
    raw = json.dumps({
        "status": "sufficient",
        "answer": "Light converts into chemical energy.",
        "sources": ["S1"],
    })
    client = FakeClient(raw)
    prepared = {
        "question": "Explain photosynthesis",
        "retrieved_chunks": [{
            "doc_id": "biology", "page_number": 1,
            "chunk_id": "biology_p1_c0", "text": "Light energy is stored as chemical energy",
        }],
        "retrieved_pages": [],
        "evidence_pages": [{"doc_id": "biology", "page_number": 1}],
        "evidence_mode": "text",
    }
    events, result = collect_generator(stream_answer(prepared, client, "gpt-6-luna"))
    assert result["status"] == "grounded"
    assert client.calls == 1
    assert all(item["type"] != "input_image" for item in client.inputs[0]["input"][1]["content"])
    assert "Light converts" in "".join(event[1]["text"] for event in events)


def test_visual_grounded_stream_includes_selected_image(tmp_path):
    image = tmp_path / "page_001.png"
    image.write_bytes(b"fake png")
    raw = json.dumps({
        "status": "sufficient",
        "answer": "The equation is supported by page 1.",
        "sources": ["S1"],
    })
    client = FakeClient(raw)
    prepared = {
        "question": "Explain the formula",
        "retrieved_chunks": [{
            "doc_id": "physics", "page_number": 1,
            "chunk_id": "physics_p1_c0", "text": "A fragmented equation",
        }],
        "retrieved_pages": [{"doc_id": "physics", "page_number": 1, "image_path": image}],
        "evidence_pages": [{"doc_id": "physics", "page_number": 1}],
        "evidence_mode": "visual",
    }
    _, result = collect_generator(stream_answer(prepared, client, "gpt-6-luna"))
    assert result["status"] == "grounded"
    images = [item for item in client.inputs[0]["input"][1]["content"] if item["type"] == "input_image"]
    assert len(images) == 1
    assert client.calls == 1

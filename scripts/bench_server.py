import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["DATABASE_URL"] = os.environ.get(
    "BENCH_DATABASE_URL",
    "postgresql://doubtdesk:doubtdesk@127.0.0.1:5432/doubtdesk_bench_test",
)
os.environ.setdefault("OPENAI_API_KEY", "bench")
os.environ.setdefault("OUTPUT_JUDGE_ENABLED", "true")
os.environ.setdefault("APP_ENV", "development")

PARSE_DELAY = float(os.environ.get("BENCH_PARSE_DELAY", "0.4"))
STREAM_DELTAS = int(os.environ.get("BENCH_STREAM_DELTAS", "20"))
STREAM_DELAY = float(os.environ.get("BENCH_STREAM_DELAY", "0.05"))
JUDGE_DELAY = float(os.environ.get("BENCH_JUDGE_DELAY", "0.4"))

from doubtdesk.answering.evidence_selection import TextEvidenceAssessment
from doubtdesk.answering.output_guardrails import JudgeVerdict
from doubtdesk.answering.schemas import ModelEvidenceResult
from doubtdesk.guardrails.paraphrasing import ContextualParaphraseResult


def _answer(input_messages):
    content = json.dumps(input_messages)
    match = re.search(r"Source alias: (S\d+), PDF page number: (\d+)", content)
    if match:
        return {"status": "sufficient", "answer": "Photosynthesis converts light energy into chemical energy.",
                "sources": [match.group(1)]}
    return {"status": "insufficient", "answer": "", "sources": []}


def _parsed(input, text_format):
    if text_format is ContextualParaphraseResult:
        question = json.loads(input[1]["content"])["current_question"]
        return {"resolved_question": question,
                "questions": [question, f"Explain {question}", f"What does {question} mean?"]}
    if text_format is TextEvidenceAssessment:
        payload = json.loads(input[1]["content"])
        return {"sufficiency_score": 0.95 if payload["retrieved_text"] else 0.05, "recommended_image_pages": []}
    if text_format is ModelEvidenceResult:
        return _answer(input)
    return None


def _stream_text(input, text_format):
    if text_format is ModelEvidenceResult:
        return json.dumps(_answer(input))
    return "This is a general knowledge answer about the question that was asked."


def _pieces(text):
    size = max(1, len(text) // STREAM_DELTAS + 1)
    return [text[i:i + size] for i in range(0, len(text), size)]


def _final():
    return SimpleNamespace(usage=None, status="completed")


class SyncStream:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        for piece in _pieces(self.text):
            time.sleep(STREAM_DELAY)
            yield SimpleNamespace(type="response.output_text.delta", delta=piece)

    def get_final_response(self):
        return _final()


class AsyncStream:
    def __init__(self, text):
        self.text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def _events(self):
        for piece in _pieces(self.text):
            await asyncio.sleep(STREAM_DELAY)
            yield SimpleNamespace(type="response.output_text.delta", delta=piece)

    def __aiter__(self):
        return self._events()

    async def get_final_response(self):
        return _final()


class SyncResponses:
    def parse(self, *, model, input, text_format, **kwargs):
        time.sleep(PARSE_DELAY)
        return SimpleNamespace(output_parsed=_parsed(input, text_format), output_text="ok", usage=None, status="completed")

    def stream(self, *, model, input, text_format=None, **kwargs):
        return SyncStream(_stream_text(input, text_format))

    def create(self, *, model, input, **kwargs):
        time.sleep(PARSE_DELAY)
        return SimpleNamespace(output_text=_stream_text(input, None), usage=None, status="completed")


class AsyncResponses:
    async def parse(self, *, model, input, text_format, **kwargs):
        await asyncio.sleep(PARSE_DELAY)
        return SimpleNamespace(output_parsed=_parsed(input, text_format), output_text="ok", usage=None, status="completed")

    def stream(self, *, model, input, text_format=None, **kwargs):
        return AsyncStream(_stream_text(input, text_format))

    async def create(self, *, model, input, **kwargs):
        await asyncio.sleep(PARSE_DELAY)
        return SimpleNamespace(output_text=_stream_text(input, None), usage=None, status="completed")


class FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.responses = SyncResponses()

    def close(self):
        return None


class FakeAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.responses = AsyncResponses()

    async def close(self):
        return None


class FakeJudgeClient:
    async def close(self):
        return None


def _verdict():
    return JudgeVerdict(passed=True, relevance=1, groundedness=1, completeness=1, safety="pass", feedback="Bench approved.")


class FakeJudge:
    def __init__(self, client=None, model_name=None, pacer=None):
        pass

    def evaluate(self, **kwargs):
        time.sleep(JUDGE_DELAY)
        return _verdict()

    async def aevaluate(self, **kwargs):
        await asyncio.sleep(JUDGE_DELAY)
        return _verdict()


def seed_users(count: int):
    from doubtdesk.auth.schemas import SignupRequest
    from doubtdesk.auth.service import register_user
    from doubtdesk.db import connection, initialize_database
    from tests.db_support import ensure_database, reset_database

    ensure_database(os.environ["DATABASE_URL"])
    initialize_database()
    reset_database(os.environ["DATABASE_URL"])
    for index in range(count):
        user = register_user(SignupRequest(username=f"bench_{index}", email=f"bench_{index}@example.com",
                                           password="bench-password-123"))
        with connection() as conn:
            conn.execute("UPDATE users SET verified_at = %s WHERE user_id = %s", (int(time.time()), user["user_id"]))


def main():
    import uvicorn
    from doubtdesk.api import main as app_module
    from doubtdesk.db import close_pool

    seed_users(int(os.environ.get("BENCH_USERS", "10")))
    close_pool()
    app_module.OpenAI = FakeOpenAI
    if hasattr(app_module, "AsyncOpenAI"):
        app_module.AsyncOpenAI = FakeAsyncOpenAI
    app_module.build_output_judge = lambda config=None: (FakeJudgeClient(), FakeJudge())
    print(f"BENCH_READY_IMPORTED {time.time()}", flush=True)
    uvicorn.run(app_module.app, host="127.0.0.1", port=int(os.environ.get("BENCH_PORT", "8011")), log_level="warning")


if __name__ == "__main__":
    main()

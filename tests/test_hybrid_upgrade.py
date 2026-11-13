import asyncio
from types import SimpleNamespace
import pytest
from doubtdesk.retrieval.hybrid import reciprocal_rank_fusion
from doubtdesk.memory.store import (
    initialize_memory_db, claim_or_check_conversation, save_exchange,
    ConversationLimitReached, delete_user_conversation,
    list_user_conversations, get_user_recent_exchanges,
)
from doubtdesk.evals.schemas import EvalsRequest


def test_rrf_merges_dense_and_bm25_without_duplicate_chunks():
    a = {"chunk_id": "a", "text": "first", "doc_id": "doc", "page_number": 1, "distance": 0.1}
    b = {"chunk_id": "b", "text": "second", "doc_id": "doc", "page_number": 2}
    c = {"chunk_id": "c", "text": "third", "doc_id": "doc", "page_number": 3}
    result = reciprocal_rank_fusion([[a, b], [b, a, c]], top_k=3)
    assert [chunk["chunk_id"] for chunk in result] == ["a", "b", "c"] or [chunk["chunk_id"] for chunk in result] == ["b", "a", "c"]
    assert len({chunk["chunk_id"] for chunk in result}) == 3
    assert result[0]["rrf_score"] > result[2]["rrf_score"]


def test_conversation_limit_preserves_existing_history_and_rejects_new(tmp_path):
    db_path = tmp_path / "conversations.db"
    initialize_memory_db(db_path)
    assert claim_or_check_conversation("first", "u", db_path, max_conversations=2)
    save_exchange("first", "first question", "first answer", [], "grounded", db_path)
    assert claim_or_check_conversation("second", "u", db_path, max_conversations=2)
    assert claim_or_check_conversation("first", "u", db_path, max_conversations=2)

    with pytest.raises(ConversationLimitReached, match="2-conversation limit"):
        claim_or_check_conversation("third", "u", db_path, max_conversations=2)

    assert {row["conversation_id"] for row in list_user_conversations("u", db_path)} == {"first", "second"}
    assert get_user_recent_exchanges("first", "u", db_path=db_path)[0]["answer"] == "first answer"
    assert delete_user_conversation("second", "u", db_path)
    assert claim_or_check_conversation("third", "u", db_path, max_conversations=2)
    assert not claim_or_check_conversation("second", "u", db_path, max_conversations=2)


def test_eval_requires_five_questions():
    item = {"question": "Explain photosynthesis", "reference_answer": "Plants use light to produce sugars", "must_include": ["Light energy is used"]}
    with pytest.raises(Exception):
        EvalsRequest(doc_ids=["doc"], questions=[item])
    valid = EvalsRequest(doc_ids=["doc"], questions=[item] * 5)
    assert len(valid.questions) == 5
    assert valid.mode == "full"


def test_judge_metric_failure_never_becomes_a_pass():
    from doubtdesk.evals.judge import RagasJudge
    from doubtdesk.evals.schemas import MetricThresholds

    class FakeMetric:
        def __init__(self, value=None, fail=False):
            self.value = value
            self.fail = fail

        async def ascore(self, **kwargs):
            if self.fail:
                raise ValueError("invalid judge result")
            return SimpleNamespace(value=self.value)

    judge = RagasJudge.__new__(RagasJudge)
    judge.metrics = {
        "context_precision": FakeMetric(0.85),
        "context_recall": FakeMetric(fail=True),
    }
    result = asyncio.run(judge.score(
        question="What is photosynthesis?",
        contexts=["Plants use light"],
        reference="Plants use light",
        requirements=["Light is needed"],
        thresholds=MetricThresholds(),
    ))
    assert result["scores"]["context_recall"] is None
    assert not result["all_passed"]
    assert result["passed"]["context_precision"]

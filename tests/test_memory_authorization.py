from psycopg.types.json import Jsonb

from doubtdesk.db import connection

from doubtdesk.memory.context import build_history
from doubtdesk.memory.store import (
    claim_or_check_conversation,
    get_user_recent_exchanges,
    save_exchange,
)


def test_conversation_claim_and_ownership():
    assert claim_or_check_conversation("chat_A", "user_A")
    assert claim_or_check_conversation("chat_A", "user_A")
    assert not claim_or_check_conversation("chat_A", "user_B")
    assert claim_or_check_conversation("chat_B", "user_B")

    with connection() as conn:
        assert conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = 'chat_A'"
        ).fetchone() == ("user_A",)


def test_legacy_conversation_cannot_be_claimed():
    with connection() as conn:
        conn.execute(
            """
            INSERT INTO exchanges (conversation_id, question, answer, doc_ids, status)
            VALUES (%s, %s, %s, %s, %s)
            """,
            ("legacy", "Old question", "Private answer", Jsonb([]), "general_knowledge"),
        )

    assert not claim_or_check_conversation("legacy", "user_A")
    assert not claim_or_check_conversation("legacy", "user_B")
    assert get_user_recent_exchanges("legacy", "user_A") == []


def test_user_history_never_crosses_accounts():
    assert claim_or_check_conversation("chat_A", "user_A")
    assert claim_or_check_conversation("chat_B", "user_B")

    save_exchange("chat_A", "Chemistry question", "Chemistry answer", ["doc_A"], "grounded", user_id="user_A")
    save_exchange("chat_B", "Physics question", "Physics answer", ["doc_B"], "grounded", user_id="user_B")

    owner = get_user_recent_exchanges("chat_A", "user_A")
    stranger = get_user_recent_exchanges("chat_A", "user_B")
    assert [entry["question"] for entry in owner] == ["Chemistry question"]
    assert stranger == []
    assert get_user_recent_exchanges("missing", "user_A") == []


def test_user_context_preserves_recency_across_document_changes():
    assert claim_or_check_conversation("chat_A", "user_A")

    save_exchange("chat_A", "Biology", "Biology explanation", ["doc_A"], "grounded", user_id="user_A")
    save_exchange("chat_A", "Unrelated physics", "Physics explanation", ["doc_B"], "grounded", user_id="user_A")
    save_exchange("chat_A", "Off topic", "Refusal", ["doc_A"], "out_of_scope", user_id="user_A")
    save_exchange("chat_A", "Chlorophyll", "Chlorophyll explanation", ["doc_A"], "grounded", user_id="user_A")

    context = build_history(
        conversation_id="chat_A",
        user_id="user_A",
        question="What about light?",
        doc_ids=["doc_A"],
    )

    assert [entry["question"] for entry in context] == ["Unrelated physics", "Chlorophyll"]
    assert context[-1]["answer_excerpt"] == "Chlorophyll explanation"
    assert build_history(
        "chat_A", "user_B", "Steal context", ["doc_A"]
    ) == []
    assert [item["question"] for item in build_history(
        "chat_A", "user_A", "New document", ["doc_X"]
    )] == ["Unrelated physics", "Chlorophyll"]

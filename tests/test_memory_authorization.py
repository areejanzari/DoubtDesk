import sqlite3
from contextlib import closing

from doubtdesk.memory.context import get_user_conversation_context
from doubtdesk.memory.store import (
    claim_or_check_conversation,
    get_user_recent_exchanges,
    initialize_memory_db,
    save_exchange,
)


def test_conversation_claim_and_ownership(tmp_path):
    db_path = tmp_path / "conversations.db"
    initialize_memory_db(db_path)

    assert claim_or_check_conversation("chat_A", "user_A", db_path)
    assert claim_or_check_conversation("chat_A", "user_A", db_path)
    assert not claim_or_check_conversation("chat_A", "user_B", db_path)
    assert claim_or_check_conversation("chat_B", "user_B", db_path)

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = 'chat_A'"
        ).fetchone() == ("user_A",)


def test_legacy_conversation_cannot_be_claimed(tmp_path):
    db_path = tmp_path / "conversations.db"
    initialize_memory_db(db_path)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            """
            INSERT INTO exchanges (conversation_id, question, answer, doc_ids, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("legacy", "Old question", "Private answer", "[]", "general_knowledge"),
        )

    assert not claim_or_check_conversation("legacy", "user_A", db_path)
    assert not claim_or_check_conversation("legacy", "user_B", db_path)
    assert get_user_recent_exchanges("legacy", "user_A", db_path=db_path) == []


def test_user_history_never_crosses_accounts(tmp_path):
    db_path = tmp_path / "conversations.db"
    initialize_memory_db(db_path)
    assert claim_or_check_conversation("chat_A", "user_A", db_path)
    assert claim_or_check_conversation("chat_B", "user_B", db_path)

    save_exchange("chat_A", "Chemistry question", "Chemistry answer", ["doc_A"], "grounded", db_path)
    save_exchange("chat_B", "Physics question", "Physics answer", ["doc_B"], "grounded", db_path)

    owner = get_user_recent_exchanges("chat_A", "user_A", db_path=db_path)
    stranger = get_user_recent_exchanges("chat_A", "user_B", db_path=db_path)
    assert [entry["question"] for entry in owner] == ["Chemistry question"]
    assert stranger == []
    assert get_user_recent_exchanges("missing", "user_A", db_path=db_path) == []


def test_user_context_preserves_filtering_and_answer_excerpt(tmp_path):
    db_path = tmp_path / "conversations.db"
    initialize_memory_db(db_path)
    assert claim_or_check_conversation("chat_A", "user_A", db_path)

    save_exchange("chat_A", "Biology", "Biology explanation", ["doc_A"], "grounded", db_path)
    save_exchange("chat_A", "Unrelated physics", "Physics explanation", ["doc_B"], "grounded", db_path)
    save_exchange("chat_A", "Off topic", "Refusal", ["doc_A"], "out_of_scope", db_path)
    save_exchange("chat_A", "Chlorophyll", "Chlorophyll explanation", ["doc_A"], "grounded", db_path)

    context = get_user_conversation_context(
        conversation_id="chat_A",
        user_id="user_A",
        question="What about light?",
        doc_ids=["doc_A"],
        db_path=db_path,
    )

    assert [entry["question"] for entry in context] == ["Biology", "Chlorophyll"]
    assert context[-1]["answer_excerpt"] == "Chlorophyll explanation"
    assert get_user_conversation_context(
        "chat_A", "user_B", "Steal context", ["doc_A"], db_path=db_path
    ) == []
    assert get_user_conversation_context(
        "chat_A", "user_A", "Wrong document", ["doc_X"], db_path=db_path
    ) == []

import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from doubtdesk.memory.store import (
    ConversationLimitReached,
    claim_or_check_conversation,
    delete_user_conversation,
    get_user_conversation,
    get_user_recent_exchanges,
    initialize_memory_db,
    list_user_conversations,
    rename_user_conversation,
    save_exchange,
)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "conversations.db"
    initialize_memory_db(path)
    return path


def test_limit_30_and_followup_allowed_without_eviction(db):
    ids = [str(uuid4()) for _ in range(30)]
    for item in ids:
        assert claim_or_check_conversation(item, "owner", db)
    with pytest.raises(ConversationLimitReached):
        claim_or_check_conversation(str(uuid4()), "owner", db)
    assert claim_or_check_conversation(ids[0], "owner", db)
    assert len(list_user_conversations("owner", db)) == 30
    assert claim_or_check_conversation(str(uuid4()), "other", db)
    assert len(list_user_conversations("owner", db)) == 30
    assert delete_user_conversation(ids[0], "owner", db)
    assert not claim_or_check_conversation(ids[0], "owner", db)
    assert claim_or_check_conversation(str(uuid4()), "owner", db)
    assert len(list_user_conversations("owner", db)) == 30


def test_concurrent_30_limit_atomic(db):
    ids = [str(uuid4()) for _ in range(60)]

    def create(item):
        try:
            return claim_or_check_conversation(item, "same", db)
        except ConversationLimitReached:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(create, ids))
    assert sum(results) == 30
    assert len(list_user_conversations("same", db)) == 30


def test_title_auto_rename_sort_and_no_cross_account_leak(db):
    a, b = str(uuid4()), str(uuid4())
    claim_or_check_conversation(a, "alice", db)
    claim_or_check_conversation(b, "bob", db)
    exchange_id = save_exchange(
        a,
        "  Explain   nephron structure? ",
        "PCT, Henle's loop, DCT",
        ["pdf_a"],
        "grounded",
        db,
        user_id="alice",
        formatted_answer="PCT, Henle's loop, DCT\nSources:\n- biology.pdf: PDF page(s) 3",
        sources=[{"doc_id": "pdf_a", "page_numbers": [3]}],
    )
    assert exchange_id > 0
    detail = get_user_conversation(a, "alice", db)
    assert detail["title"] == "Explain nephron structure?"
    assert detail["messages"][0]["sources"] == [{"doc_id": "pdf_a", "page_numbers": [3]}]
    assert detail["messages"][0]["formatted_answer"].endswith("PDF page(s) 3")
    assert get_user_conversation(a, "bob", db) is None
    assert not rename_user_conversation(a, "bob", "Steal chat", db)
    assert not delete_user_conversation(a, "bob", db)
    assert rename_user_conversation(a, "alice", "Kidney notes", db)
    save_exchange(a, "Next question", "Answer", [], "general_knowledge", db, user_id="alice")
    assert get_user_conversation(a, "alice", db)["title"] == "Kidney notes"
    assert get_user_recent_exchanges(a, "bob", db_path=db) == []


def test_deletion_removes_messages_without_reclaim(db):
    conversation_id = str(uuid4())
    claim_or_check_conversation(conversation_id, "alice", db)
    save_exchange(conversation_id, "Q", "A", [], "grounded", db, user_id="alice")
    assert delete_user_conversation(conversation_id, "alice", db)
    assert get_user_conversation(conversation_id, "alice", db) is None
    assert not claim_or_check_conversation(conversation_id, "alice", db)
    with pytest.raises(ValueError, match="Conversation not found"):
        save_exchange(conversation_id, "Q", "A", [], "grounded", db, user_id="alice")
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM exchanges").fetchone()[0] == 0


def test_migration_preserves_old_exchanges_and_creates_title(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_accessed_ns INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE exchanges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                doc_ids TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("INSERT INTO conversations(conversation_id, user_id) VALUES (?, ?)", ("old", "alice"))
        conn.execute(
            "INSERT INTO exchanges(conversation_id, question, answer, doc_ids, status) VALUES (?, ?, ?, ?, ?)",
            ("old", "Old biology question", "Old answer", "[]", "grounded"),
        )
    initialize_memory_db(path)
    initialize_memory_db(path)
    detail = get_user_conversation("old", "alice", path)
    assert detail["title"] == "Old biology question"
    assert detail["messages"][0]["answer"] == "Old answer"
    assert detail["messages"][0]["sources"] == []


def test_api_routes_ownership_limit_crud(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_memory_db(Path("data/conversations.db"))
    from doubtdesk.auth.dependencies import require_current_user
    from doubtdesk.conversations import routes
    delete_memories = MagicMock()
    monkeypatch.setattr(routes, "delete_session_memories", delete_memories)

    user = {"user_id": "alice"}
    app = FastAPI()
    app.include_router(routes.router)
    app.state.memory_collection = MagicMock()
    app.dependency_overrides[require_current_user] = lambda: user
    with TestClient(app) as client:
        assert client.get("/conversations").json() == []
        created = client.post("/conversations")
        assert created.status_code == 201
        conversation_id = created.json()["conversation_id"]
        assert created.json()["title"] == "New chat"
        assert client.get(f"/conversations/{conversation_id}").status_code == 200
        renamed = client.patch(f"/conversations/{conversation_id}", json={"title": "Biology Q&A"})
        assert renamed.status_code == 200
        assert renamed.json()["title"] == "Biology Q&A"
        user["user_id"] = "bob"
        assert client.get(f"/conversations/{conversation_id}").status_code == 404
        assert client.patch(f"/conversations/{conversation_id}", json={"title": "Hijack"}).status_code == 404
        assert client.delete(f"/conversations/{conversation_id}").status_code == 404
        user["user_id"] = "alice"
        for _ in range(29):
            assert client.post("/conversations").status_code == 201
        assert client.post("/conversations").status_code == 409
        assert len(client.get("/conversations").json()) == 30
        assert client.delete(f"/conversations/{conversation_id}").status_code == 204
        delete_memories.assert_called_once()
        assert client.post("/conversations").status_code == 201


def test_ask_and_stream_reject_31st_before_model_calls(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_memory_db(Path("data/conversations.db"))
    from doubtdesk.ingestion.registry import initialize_registry
    from doubtdesk.auth.dependencies import require_current_user
    from doubtdesk.api import routes
    initialize_registry(Path("data/documents.db"))
    for _ in range(30):
        claim_or_check_conversation(str(uuid4()), "alice")

    stream = MagicMock()
    monkeypatch.setattr(routes, "stream_answer", stream)
    app = FastAPI()
    app.include_router(routes.router)
    app.state.graph = MagicMock()
    app.state.preparation_graph = MagicMock()
    app.dependency_overrides[require_current_user] = lambda: {"user_id": "alice"}
    with TestClient(app) as client:
        payload = {"question": "What is a nephron?", "doc_ids": []}
        for endpoint in ["/ask", "/ask/stream"]:
            response = client.post(endpoint, json=payload)
            assert response.status_code == 409
            assert "30-conversation limit" in response.json()["detail"]
        app.state.graph.invoke.assert_not_called()
        app.state.preparation_graph.invoke.assert_not_called()
    stream.assert_not_called()


def test_completed_ask_and_stream_save_history(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initialize_memory_db(Path("data/conversations.db"))
    from doubtdesk.ingestion.registry import initialize_registry
    from doubtdesk.auth.dependencies import require_current_user
    from doubtdesk.api import routes
    initialize_registry(Path("data/documents.db"))

    def stream_answer(*args, **kwargs):
        yield ("token", {"text": "The answer"})
        return {
            "status": "grounded",
            "answer": "The answer",
            "sources": [{"doc_id": "pdf", "page_numbers": [2]}],
        }

    monkeypatch.setattr(routes, "add_exchange_memory", MagicMock())
    monkeypatch.setattr(routes, "stream_answer", stream_answer)
    monkeypatch.setattr(routes, "finalize_stream_result", lambda result, **kwargs: {
        **result,
        "formatted_answer": "The answer\n\nSources:\n- biology.pdf: PDF page(s) 2",
    })
    app = FastAPI()
    app.include_router(routes.router)
    app.state.graph = MagicMock()
    app.state.graph.invoke.return_value = {
        "status": "general_knowledge",
        "answer": "Gravity explanation",
        "sources": [],
        "formatted_answer": "Gravity explanation\n\nNot from document.",
    }
    app.state.preparation_graph = MagicMock()
    app.state.preparation_graph.invoke.return_value = {"question": "A biology question"}
    app.state.llm_client = MagicMock()
    app.state.model_name = "fake-model"
    app.state.output_judge = MagicMock()
    app.state.embedding_model = MagicMock()
    app.state.memory_collection = MagicMock()
    app.dependency_overrides[require_current_user] = lambda: {"user_id": "alice"}
    with TestClient(app) as client:
        first = client.post("/ask", json={"question": "Explain gravity", "doc_ids": []})
        assert first.status_code == 200
        cid = first.json()["conversation_id"]
        detail = get_user_conversation(cid, "alice")
        assert detail["title"] == "Explain gravity"
        assert detail["messages"][0]["formatted_answer"].endswith("Not from document.")

        stream = client.post(
            "/ask/stream",
            json={"question": "A biology question", "doc_ids": [], "conversation_id": cid},
        )
        assert stream.status_code == 200
        assert "event: complete" in stream.text
        detail = get_user_conversation(cid, "alice")
        assert [item["question"] for item in detail["messages"]] == [
            "Explain gravity", "A biology question"
        ]
        assert detail["messages"][1]["sources"] == [{"doc_id": "pdf", "page_numbers": [2]}]
        assert detail["title"] == "Explain gravity"


def test_router_modules_remain_canonical():
    from doubtdesk.api import routes, streaming
    from doubtdesk.conversations import routes as conversation_routes
    from doubtdesk.memory import vector

    assert routes is sys.modules["doubtdesk.api.routes"]
    assert streaming is sys.modules["doubtdesk.api.streaming"]
    assert conversation_routes is sys.modules["doubtdesk.conversations.routes"]
    assert vector is sys.modules["doubtdesk.memory.vector"]
    assert routes.stream_answer is streaming.stream_answer
    assert routes.finalize_stream_result is streaming.finalize_stream_result
    assert conversation_routes.delete_session_memories is vector.delete_session_memories

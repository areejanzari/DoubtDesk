import sqlite3
from contextlib import closing
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pymupdf
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from doubtdesk.api.main import app
from doubtdesk.auth.dependencies import require_current_user
from doubtdesk.ingestion.registry import (
    attach_document_to_user,
    initialize_registry,
    register_document,
    update_document_status,
)
from doubtdesk.memory.store import initialize_memory_db


@pytest.fixture
def account():
    return {"user_id": "user_A"}


@pytest.fixture
def client(tmp_path, monkeypatch, account):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    initialize_registry(tmp_path / "data/documents.db")
    initialize_memory_db(tmp_path / "data/conversations.db")

    app.state.graph = MagicMock()
    app.state.embedding_model = MagicMock()
    app.state.collection = MagicMock()

    previous_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[require_current_user] = lambda: account

    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        test_client.close()
        app.dependency_overrides.clear()
        app.dependency_overrides.update(previous_overrides)


def add_completed_document(tmp_path, filename="biology.pdf", owners=("user_A",)):
    doc_id = str(uuid4())
    db_path = tmp_path / "data/documents.db"
    register_document(doc_id, filename, f"hash_{doc_id}", db_path)
    update_document_status(doc_id, "completed", 2, db_path)
    for owner in owners:
        attach_document_to_user(owner, doc_id, filename, db_path)
    return doc_id


def set_graph_answer():
    app.state.graph.invoke.return_value = {
        "status": "general_knowledge",
        "answer": "Gravity attracts masses.",
        "sources": [],
        "formatted_answer": "Gravity attracts masses.\n\nNot from document.",
    }


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"doc_ids": []},
        {"question": "Explain gravity", "doc_ids": "invalid"},
    ],
)
def test_invalid_ask_request(client, payload):
    response = client.post("/ask", json=payload)
    assert response.status_code == 422


def test_ask_without_documents(client):
    set_graph_answer()

    response = client.post(
        "/ask",
        json={"question": "What is gravity?", "doc_ids": [], "conversation_id": None},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "general_knowledge"
    assert response.json()["sources"] == []
    app.state.graph.invoke.assert_called_once()

    state = app.state.graph.invoke.call_args.args[0]
    assert state["user_id"] == "user_A"
    assert state["question"] == "What is gravity?"
    assert state["doc_ids"] == []
    assert state["conversation_id"] == response.json()["conversation_id"]

    with closing(sqlite3.connect("data/conversations.db")) as conn:
        row = conn.execute(
            "SELECT user_id FROM conversations WHERE conversation_id = ?",
            (state["conversation_id"],),
        ).fetchone()
        saved = conn.execute(
            "SELECT question FROM exchanges WHERE conversation_id = ?",
            (state["conversation_id"],),
        ).fetchone()

    assert row == ("user_A",)
    assert saved == ("What is gravity?",)


def test_ask_with_owned_pdf(client, tmp_path):
    doc_id = add_completed_document(tmp_path)
    set_graph_answer()

    response = client.post(
        "/ask",
        json={"question": "Explain this PDF", "doc_ids": [doc_id]},
    )

    assert response.status_code == 200
    state = app.state.graph.invoke.call_args.args[0]
    assert state["user_id"] == "user_A"
    assert state["doc_ids"] == [doc_id]


def test_ask_rejects_other_users_pdf(client, tmp_path):
    doc_id = add_completed_document(tmp_path, owners=("user_B",))
    response = client.post(
        "/ask",
        json={"question": "What does this say?", "doc_ids": [doc_id]},
    )
    assert response.status_code == 400
    app.state.graph.invoke.assert_not_called()


def test_ask_rejects_mixed_document_access(client, tmp_path):
    owned = add_completed_document(tmp_path, "biology.pdf", ("user_A",))
    private = add_completed_document(tmp_path, "physics.pdf", ("user_B",))
    response = client.post(
        "/ask",
        json={"question": "Compare", "doc_ids": [owned, private]},
    )
    assert response.status_code == 400
    app.state.graph.invoke.assert_not_called()


def test_conversation_followup_and_cross_user_denial(client, account):
    set_graph_answer()
    first = client.post(
        "/ask", json={"question": "What is gravity?", "doc_ids": []}
    )
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]

    followup = client.post(
        "/ask",
        json={
            "question": "And mass?",
            "doc_ids": [],
            "conversation_id": conversation_id,
        },
    )
    assert followup.status_code == 200
    assert app.state.graph.invoke.call_count == 2

    account["user_id"] = "user_B"
    denied = client.post(
        "/ask",
        json={
            "question": "Read earlier messages",
            "doc_ids": [],
            "conversation_id": conversation_id,
        },
    )
    assert denied.status_code == 404
    assert denied.json()["detail"] == "Conversation not found."
    assert app.state.graph.invoke.call_count == 2


def test_legacy_conversation_cannot_be_claimed(client):
    conversation_id = str(uuid4())
    with closing(sqlite3.connect("data/conversations.db")) as conn, conn:
        conn.execute(
            """
            INSERT INTO exchanges (conversation_id, question, answer, doc_ids, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (conversation_id, "Legacy question", "Legacy answer", "[]", "general_knowledge"),
        )

    response = client.post(
        "/ask",
        json={
            "question": "Follow up",
            "doc_ids": [],
            "conversation_id": conversation_id,
        },
    )
    assert response.status_code == 404
    app.state.graph.invoke.assert_not_called()


def test_ask_requires_user_dependency(client):
    def no_user():
        raise HTTPException(status_code=401, detail="Not authenticated")

    app.dependency_overrides[require_current_user] = no_user
    response = client.post(
        "/ask", json={"question": "Explain gravity", "doc_ids": []}
    )
    assert response.status_code == 401
    app.state.graph.invoke.assert_not_called()


def test_list_documents(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_isolated_by_user(client, tmp_path, account):
    shared_id = add_completed_document(
        tmp_path, "shared.pdf", ("user_A", "user_B")
    )
    private_id = add_completed_document(tmp_path, "private.pdf", ("user_A",))

    result_a = client.get("/documents")
    assert result_a.status_code == 200
    assert {item["doc_id"] for item in result_a.json()} == {shared_id, private_id}

    account["user_id"] = "user_B"
    result_b = client.get("/documents")
    assert result_b.status_code == 200
    assert {item["doc_id"] for item in result_b.json()} == {shared_id}


@pytest.mark.parametrize(
    "filename,data,mime,detail",
    [
        ("notes.txt", b"Hello", "text/plain", "Only pdf files are supported."),
        ("empty.pdf", b"", "application/pdf", "Uploaded PDF is empty."),
        ("fake.pdf", b"Not a real PDF", "application/pdf", "Invalid PDF file content."),
        ("broken.pdf", b"%PDF-corrupted", "application/pdf", "PDF is corrupted or unreadable."),
    ],
)
def test_reject_bad_pdf(client, filename, data, mime, detail):
    response = client.post("/documents/upload", files={"file": (filename, data, mime)})
    assert response.status_code == 400
    assert response.json()["detail"] == detail


def test_successful_pdf_upload(client):
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf_bytes = pdf.tobytes()

    doc_id = str(uuid4())
    with patch(
        "doubtdesk.api.routes.process_pdf",
        return_value={"doc_id": doc_id, "status": "completed", "page_count": 1},
    ) as mock_process:
        response = client.post(
            "/documents/upload",
            files={"file": ("biology.pdf", pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": doc_id,
        "status": "completed",
        "page_count": 1,
        "filename": "biology.pdf",
    }
    mock_process.assert_called_once()
    assert mock_process.call_args.kwargs["user_id"] == "user_A"
    assert mock_process.call_args.args[0].name == "biology.pdf"


def test_invalid_document_id(client):
    response = client.get("/documents/invalid-id/pages/1/image")
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid document ID."


def test_invalid_page_number(client):
    response = client.get(f"/documents/{uuid4()}/pages/0/image")
    assert response.status_code == 400
    assert response.json()["detail"] == "Page number must be greater than zero."


def test_document_not_found(client):
    response = client.get(f"/documents/{uuid4()}/pages/1/image")
    assert response.status_code == 404
    assert response.json()["detail"] == "Document or page not found."


def test_page_image_access_isolated_by_user(client, tmp_path, account):
    doc_id = add_completed_document(tmp_path, owners=("user_A",))
    image_path = tmp_path / "data/page_images" / doc_id / "page_001.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"test-image-bytes")

    own = client.get(f"/documents/{doc_id}/pages/1/image")
    assert own.status_code == 200
    assert own.content == b"test-image-bytes"

    out_of_range = client.get(f"/documents/{doc_id}/pages/3/image")
    assert out_of_range.status_code == 404

    account["user_id"] = "user_B"
    denied = client.get(f"/documents/{doc_id}/pages/1/image")
    assert denied.status_code == 404


def test_shared_page_image_access(client, tmp_path, account):
    doc_id = add_completed_document(tmp_path, owners=("user_A", "user_B"))
    image_path = tmp_path / "data/page_images" / doc_id / "page_001.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"shared-image")

    account["user_id"] = "user_B"
    response = client.get(f"/documents/{doc_id}/pages/1/image")
    assert response.status_code == 200
    assert response.content == b"shared-image"

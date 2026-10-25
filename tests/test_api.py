import sqlite3
from unittest.mock import MagicMock, patch
from uuid import uuid4
import pymupdf
import pytest
from fastapi.testclient import TestClient
from doubtdesk.api.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "data").mkdir()

    with sqlite3.connect(tmp_path / "data/documents.db") as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                page_count INTEGER NOT NULL DEFAULT 0
            )
        """)

    app.state.graph = MagicMock()

    return TestClient(app)

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
    app.state.graph.invoke.return_value = {
        "status": "general_knowledge",
        "answer": "Gravity attracts masses.",
        "sources": [],
        "formatted_answer": (
            "Gravity attracts masses.\n\nNot from document."
        ),
    }

    response = client.post(
        "/ask",
        json={"question": "What is gravity?", "doc_ids": []},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "general_knowledge"
    assert response.json()["sources"] == []

    app.state.graph.invoke.assert_called_once_with({
        "question": "What is gravity?",
        "doc_ids": [],
    })

def test_list_documents(client):
    response = client.get("/documents")

    assert response.status_code == 200
    assert response.json() == []

def test_reject_non_pdf(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": ("notes.txt", b"Hello", "text/plain"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only pdf files are supported."


def test_reject_empty_pdf(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": ("empty.pdf", b"", "application/pdf"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded PDF is empty."


def test_reject_invalid_pdf_signature(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": ("fake.pdf", b"Not a real PDF", "application/pdf"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid PDF file content."


def test_reject_corrupted_pdf(client):
    response = client.post(
        "/documents/upload",
        files={
            "file": ("broken.pdf", b"%PDF-corrupted", "application/pdf"),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "PDF is corrupted or unreadable."

def test_successful_pdf_upload(client):
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf_bytes = pdf.tobytes()

    doc_id = str(uuid4())

    with (
        patch(
            "doubtdesk.api.routes.process_pdf",
            return_value={
                "doc_id": doc_id,
                "status": "completed",
                "page_count": 1,
            },
        ) as mock_process,
        patch(
            "doubtdesk.api.routes.get_document_names",
            return_value={doc_id: "biology.pdf"},
        ),
    ):
        response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "biology.pdf",
                    pdf_bytes,
                    "application/pdf",
                ),
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "doc_id": doc_id,
        "status": "completed",
        "page_count": 1,
        "filename": "biology.pdf",
    }

    mock_process.assert_called_once()

def test_invalid_document_id(client):
    response = client.get(
        "/documents/invalid-id/pages/1/image"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid document ID."


def test_invalid_page_number(client):
    doc_id = str(uuid4())

    response = client.get(
        f"/documents/{doc_id}/pages/0/image"
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Page number must be greater than zero."
    )


def test_document_not_found(client):
    doc_id = str(uuid4())

    response = client.get(
        f"/documents/{doc_id}/pages/1/image"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Document not found."


import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from doubtdesk.db import connection
from doubtdesk.ingestion import jobs
from doubtdesk.ingestion.registry import (
    STALE_PROCESSING_SECONDS,
    DocumentProcessing,
    claim_document_ingestion,
    fail_stale_ingestions,
    get_document_by_hash,
    get_user_document_status,
    list_documents,
    register_document,
    update_document_status,
)
from doubtdesk.ingestion.service import run_ingestion, start_ingestion
from doubtdesk.storage.images import existing_page_numbers


def status_for(user_id, doc_id):
    with connection() as conn:
        return get_user_document_status(conn, user_id, doc_id)


def listed(user_id):
    with connection() as conn:
        return [item["doc_id"] for item in list_documents(conn, user_id=user_id)]


@pytest.fixture
def pdf_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "upload.pdf"
    path.write_bytes(b"%PDF-1.4 unique " + uuid4().bytes)
    return path


def test_new_upload_is_attached_as_processing_and_queued(pdf_file):
    result = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    assert result["status"] == "processing" and result["queued"] is True
    assert status_for("alice", result["doc_id"])["status"] == "processing"
    assert listed("alice") == []


def test_second_uploader_joins_in_flight_ingestion_without_new_job(pdf_file):
    first = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    second = start_ingestion(pdf_file, filename="my-bio.pdf", user_id="bob")
    assert second == {"doc_id": first["doc_id"], "status": "processing", "page_count": 0, "queued": False}
    assert status_for("bob", first["doc_id"])["status"] == "processing"


def test_run_ingestion_completes_document_and_removes_upload(pdf_file):
    started = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    model = MagicMock()
    with (
        patch("doubtdesk.ingestion.service.ingest_document", return_value=["p1", "p2", "p3"]) as ingest,
        patch("doubtdesk.ingestion.service.chunk_document", return_value=["c1"]),
        patch("doubtdesk.ingestion.service.embed_chunks", return_value=np.zeros((1, 384))) as embed,
        patch("doubtdesk.ingestion.service.replace_document_chunks") as replace,
    ):
        run_ingestion(started["doc_id"], pdf_file, model)
    assert embed.call_args.args[1] is model
    replace.assert_called_once()
    assert callable(ingest.call_args.kwargs["on_page"])
    assert status_for("alice", started["doc_id"]) == {"doc_id": started["doc_id"], "status": "completed", "page_count": 3, "unreadable_pages": 0}
    assert listed("alice") == [started["doc_id"]]
    assert not pdf_file.exists()


def test_run_ingestion_failure_marks_failed_and_removes_upload(pdf_file):
    started = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    with patch("doubtdesk.ingestion.service.ingest_document", side_effect=RuntimeError("OCR crashed")):
        run_ingestion(started["doc_id"], pdf_file, MagicMock())
    assert status_for("alice", started["doc_id"])["status"] == "failed"
    assert listed("alice") == []
    assert not pdf_file.exists()


def test_failed_document_can_be_uploaded_again(pdf_file):
    started = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    update_document_status(started["doc_id"], "failed", 0)
    retried = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    assert retried["doc_id"] == started["doc_id"] and retried["queued"] is True


def test_completed_document_with_artifacts_is_reused(pdf_file):
    started = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    update_document_status(started["doc_id"], "completed", 2)
    with patch("doubtdesk.ingestion.service.cached_artifacts_exist", return_value=True):
        result = start_ingestion(pdf_file, filename="mine.pdf", user_id="bob")
    assert result == {"doc_id": started["doc_id"], "status": "cached", "page_count": 2, "queued": False}
    assert listed("bob") == [started["doc_id"]]


def set_started_at(doc_id, value):
    with connection() as conn:
        conn.execute("UPDATE documents SET processing_started_at = %s WHERE doc_id = %s", (value, doc_id))


def test_stale_processing_claim_is_reclaimable_but_fresh_one_is_not():
    claim_document_ingestion("doc-stale", "a.pdf", "hash-stale")
    with pytest.raises(DocumentProcessing):
        claim_document_ingestion("other", "a.pdf", "hash-stale")
    set_started_at("doc-stale", int(time.time()) - STALE_PROCESSING_SECONDS - 5)
    assert claim_document_ingestion("other", "a.pdf", "hash-stale") == {"doc_id": "doc-stale", "status": "processing", "page_count": 0}


def test_startup_recovery_fails_only_stale_ingestions():
    claim_document_ingestion("fresh", "a.pdf", "hash-fresh")
    claim_document_ingestion("old", "b.pdf", "hash-old")
    register_document("legacy", "c.pdf", "hash-legacy")
    update_document_status("legacy", "processing", 0)
    set_started_at("old", int(time.time()) - STALE_PROCESSING_SECONDS - 5)
    assert fail_stale_ingestions() == 2
    assert get_document_by_hash("hash-fresh")["status"] == "processing"
    assert get_document_by_hash("hash-old")["status"] == "failed"
    assert get_document_by_hash("hash-legacy")["status"] == "failed"


def test_existing_page_numbers_reads_local_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    directory = tmp_path / "data/page_images/doc1"
    directory.mkdir(parents=True)
    for name in ["page_001.png", "page_003.png", "page_1000.png", "notes.txt"]:
        (directory / name).write_bytes(b"x")
    assert existing_page_numbers("doc1") == {1, 3, 1000}
    assert existing_page_numbers("missing") == set()


def test_remove_stale_uploads_keeps_recent_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uploads = tmp_path / "data/uploads"
    uploads.mkdir(parents=True)
    old, fresh = uploads / "old.pdf", uploads / "fresh.pdf"
    old.write_bytes(b"x")
    fresh.write_bytes(b"x")
    past = time.time() - STALE_PROCESSING_SECONDS - 5
    os.utime(old, (past, past))
    assert jobs.remove_stale_uploads() == 1
    assert not old.exists() and fresh.exists()


def test_jobs_submit_runs_in_bounded_worker_and_shutdown_resets():
    try:
        future = jobs.submit(lambda value: value * 2, 21)
        assert future.result(timeout=5) == 42
        assert jobs._executor._max_workers == 1
    finally:
        jobs.shutdown()
    assert jobs._executor is None


@pytest.fixture
def status_client(tmp_path, monkeypatch):
    from doubtdesk.api import routes
    from doubtdesk.auth.dependencies import require_current_user
    monkeypatch.chdir(tmp_path)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[require_current_user] = lambda: {"user_id": "alice"}
    with TestClient(app) as client:
        yield client


def test_document_status_endpoint(status_client, pdf_file):
    started = start_ingestion(pdf_file, filename="biology.pdf", user_id="alice")
    response = status_client.get(f"/documents/{started['doc_id']}/status")
    assert response.status_code == 200
    assert response.json() == {"doc_id": started["doc_id"], "status": "processing", "page_count": 0, "unreadable_pages": 0}
    assert status_client.get("/documents/not-a-uuid/status").status_code == 400
    assert status_client.get(f"/documents/{uuid4()}/status").status_code == 404


def test_document_status_hidden_from_other_users(status_client, tmp_path):
    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 other " + uuid4().bytes)
    started = start_ingestion(other, filename="private.pdf", user_id="bob")
    assert status_client.get(f"/documents/{started['doc_id']}/status").status_code == 404


def test_upload_end_to_end_with_real_background_worker(status_client, monkeypatch):
    import pymupdf
    from doubtdesk.api import routes
    finished = []

    def fake_run(doc_id, path, model):
        update_document_status(doc_id, "completed", 1)
        Path(path).unlink()
        finished.append(doc_id)

    monkeypatch.setattr(routes, "run_ingestion", fake_run)
    status_client.app.state.embedding_model = MagicMock()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), str(uuid4()))
        payload = pdf.tobytes()
    try:
        response = status_client.post("/documents/upload", files={"file": ("bio.pdf", payload, "application/pdf")})
        assert response.status_code == 202
        doc_id = response.json()["doc_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and status_client.get(f"/documents/{doc_id}/status").json()["status"] != "completed":
            time.sleep(0.05)
        assert finished == [doc_id]
        assert status_client.get(f"/documents/{doc_id}/status").json()["status"] == "completed"
        assert not list(Path("data/uploads").glob("*.pdf"))
    finally:
        jobs.shutdown()

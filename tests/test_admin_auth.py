import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from doubtdesk.analytics.routes import router


@pytest.fixture
def admin_client(monkeypatch):
    monkeypatch.setenv(
        "DOUBTDESK_ADMIN_KEY",
        "test-secret-key",
    )

    test_app = FastAPI()
    test_app.include_router(router)

    with TestClient(test_app) as client:
        yield client


def test_admin_without_key(admin_client):
    response = admin_client.get("/admin/health")

    assert response.status_code == 401


def test_admin_wrong_key(admin_client):
    response = admin_client.get(
        "/admin/health",
        headers={"X-Admin-Key": "wrong-key"},
    )

    assert response.status_code == 401


def test_admin_correct_key(admin_client):
    response = admin_client.get(
        "/admin/health",
        headers={"X-Admin-Key": "test-secret-key"},
    )

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_admin_key_not_configured(admin_client, monkeypatch):
    monkeypatch.delenv("DOUBTDESK_ADMIN_KEY")

    response = admin_client.get(
        "/admin/health",
        headers={"X-Admin-Key": "test-secret-key"},
    )

    assert response.status_code == 503
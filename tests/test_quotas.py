import time
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from doubtdesk import quotas
from doubtdesk.api import routes
from doubtdesk.analytics.routes import router as admin_router
from doubtdesk.auth.dependencies import require_current_user
from doubtdesk.db import connection


def _insert_user(plan="free"):
    user_id = str(uuid4())
    with connection() as conn:
        conn.execute(
            """INSERT INTO users (user_id, username, email, password_hash, verified_at, created_at, plan)
               VALUES (%s, %s, %s, 'x', 1, 1, %s)""",
            (user_id, f"user_{user_id[:8]}", f"{user_id[:8]}@example.com", plan),
        )
    return user_id


def _ledger(user_id, status="completed", age=0, tokens=None):
    request_id = str(uuid4())
    with connection() as conn:
        conn.execute(
            "INSERT INTO ask_requests (request_id, user_id, status, started_at) VALUES (%s, %s, %s, %s)",
            (request_id, user_id, status, int(time.time()) - age),
        )
        if tokens is not None:
            conn.execute(
                """INSERT INTO ai_usage (request_id, operation, model, total_tokens, status)
                   VALUES (%s, 'general_answer', 'm', %s, 'success')""",
                (request_id, tokens),
            )
    return request_id


def _status(request_id):
    with connection() as conn:
        row = conn.execute("SELECT status FROM ask_requests WHERE request_id = %s", (request_id,)).fetchone()
    return row[0] if row else None


@pytest.fixture
def limits(monkeypatch):
    monkeypatch.setenv("FREE_DAILY_QUESTIONS", "2")
    monkeypatch.setenv("FREE_DAILY_TOKENS", "1000")
    monkeypatch.setenv("FREE_MAX_CONCURRENT", "1")
    monkeypatch.setenv("PRO_DAILY_QUESTIONS", "5")
    monkeypatch.setenv("PRO_DAILY_TOKENS", "5000")
    monkeypatch.setenv("PRO_MAX_CONCURRENT", "2")
    monkeypatch.setenv("INFLIGHT_STALE_SECONDS", "600")


def test_plan_limits_by_tier(limits):
    assert quotas.plan_limits("free") == quotas.PlanLimits(2, 1000, 1)
    assert quotas.plan_limits("pro") == quotas.PlanLimits(5, 5000, 2)
    assert quotas.plan_limits("unlimited") == quotas.PlanLimits(None, None, None)
    assert quotas.plan_limits("unknown") == quotas.plan_limits("free")


def test_question_limit_blocks_until_window_rolls(limits):
    _ledger("u")
    _ledger("u", age=3600)
    _ledger("u", age=quotas.WINDOW_SECONDS + 10)
    with pytest.raises(quotas.QuotaExceeded) as exc:
        quotas.acquire_ask_slot("u", "free", str(uuid4()))
    assert exc.value.reason == "questions"
    assert quotas.WINDOW_SECONDS - 3700 < exc.value.retry_after <= quotas.WINDOW_SECONDS - 3600
    quotas.acquire_ask_slot("u", "pro", str(uuid4()))


def test_token_budget_blocks(limits):
    _ledger("u", tokens=999)
    rid = str(uuid4())
    quotas.acquire_ask_slot("u", "free", rid)
    quotas.release_ask_slot(rid, "completed")
    with connection() as conn:
        conn.execute("DELETE FROM ask_requests WHERE request_id = %s", (rid,))
        conn.execute("UPDATE ai_usage SET total_tokens = 1000")
    with pytest.raises(quotas.QuotaExceeded) as exc:
        quotas.acquire_ask_slot("u", "free", str(uuid4()))
    assert exc.value.reason == "tokens"


def test_concurrency_cap_ignores_stale_rows(limits):
    _ledger("u", status="in_flight", age=601)
    first = str(uuid4())
    quotas.acquire_ask_slot("u", "free", first)
    with pytest.raises(quotas.QuotaExceeded) as exc:
        quotas.acquire_ask_slot("u", "free", str(uuid4()))
    assert exc.value.reason == "concurrency"
    assert exc.value.retry_after == quotas.CONCURRENCY_RETRY_SECONDS


def test_pro_concurrency(limits):
    quotas.acquire_ask_slot("u", "pro", str(uuid4()))
    quotas.acquire_ask_slot("u", "pro", str(uuid4()))
    with pytest.raises(quotas.QuotaExceeded) as exc:
        quotas.acquire_ask_slot("u", "pro", str(uuid4()))
    assert exc.value.reason == "concurrency"


def test_unlimited_is_never_blocked_but_is_recorded(limits):
    for _ in range(10):
        quotas.acquire_ask_slot("u", "unlimited", str(uuid4()))
    usage = quotas.get_user_usage("u", "unlimited")
    assert usage["questions_used"] == 10
    assert usage["in_flight"] == 10
    assert usage["questions_limit"] is None


def test_release_is_idempotent(limits):
    rid = str(uuid4())
    quotas.acquire_ask_slot("u", "free", rid)
    quotas.release_ask_slot(rid, "completed")
    quotas.release_ask_slot(rid, "failed")
    assert _status(rid) == "completed"


def test_users_from_other_accounts_do_not_share_quota(limits):
    _ledger("a")
    _ledger("a")
    quotas.acquire_ask_slot("b", "free", str(uuid4()))


def test_cleanup_removes_old_rows(limits):
    old = _ledger("u", age=quotas.RETENTION_SECONDS + 5)
    recent = _ledger("u")
    assert quotas.cleanup_ask_requests() == 1
    assert _status(old) is None
    assert _status(recent) == "completed"


def test_new_users_default_to_free_and_session_exposes_plan():
    from doubtdesk.auth import store
    user_id = _insert_user()
    token, _ = store.create_session(user_id)
    assert store.get_session(token)["plan"] == "free"
    assert quotas.set_user_plan(user_id, "pro")
    assert store.get_session(token)["plan"] == "pro"
    assert not quotas.set_user_plan(str(uuid4()), "pro")


class _Graph:
    def __init__(self, error=None):
        self.error = error

    async def ainvoke(self, state, config=None):
        if self.error:
            raise self.error
        return {"status": "general_knowledge", "answer": "A", "sources": [], "formatted_answer": "A"}


class _Prep:
    def __init__(self, error=None):
        self.error = error

    async def ainvoke(self, state, config=None):
        if self.error:
            raise self.error
        return {**state, "is_off_topic": True, "answer": "Off topic."}


def _client(monkeypatch, plan="free", graph=None, prep=None):
    monkeypatch.setattr(routes, "validate_user_document_ids", lambda *a, **kw: None)
    monkeypatch.setattr(routes, "claim_or_check_conversation", lambda **kw: True)
    monkeypatch.setattr(routes, "_save_exchange_and_memory", Mock())
    monkeypatch.setattr(routes, "graph_config", lambda: None)
    app = FastAPI()
    app.include_router(routes.router)
    app.state.graph = graph or _Graph()
    app.state.preparation_graph = prep or _Prep()
    app.state.llm_client = Mock()
    app.state.model_name = "m"
    app.state.output_judge = None
    app.dependency_overrides[require_current_user] = lambda: {"user_id": "u", "plan": plan}
    return TestClient(app)


def _ledger_statuses():
    with connection() as conn:
        return [row[0] for row in conn.execute("SELECT status FROM ask_requests ORDER BY started_at, request_id")]


def test_ask_returns_429_with_retry_after(limits, monkeypatch):
    client = _client(monkeypatch)
    for _ in range(2):
        assert client.post("/ask", json={"question": "Why?"}).status_code == 200
    response = client.post("/ask", json={"question": "Why?"})
    assert response.status_code == 429
    assert response.json()["detail"] == routes.QUOTA_MESSAGES["questions"]
    assert int(response.headers["Retry-After"]) > 0
    assert _ledger_statuses() == ["completed", "completed"]


def test_ask_failure_releases_slot(limits, monkeypatch):
    monkeypatch.setattr(routes, "delete_empty_conversation", Mock())
    client = _client(monkeypatch, graph=_Graph(RuntimeError("boom")))
    assert client.post("/ask", json={"question": "Why?"}).status_code == 500
    assert _ledger_statuses() == ["failed"]


def test_conversation_rejection_releases_slot(limits, monkeypatch):
    client = _client(monkeypatch)
    monkeypatch.setattr(routes, "claim_or_check_conversation", lambda **kw: False)
    assert client.post("/ask", json={"question": "Why?", "conversation_id": str(uuid4())}).status_code == 404
    assert _ledger_statuses() == ["failed"]


def test_stream_completes_and_releases(limits, monkeypatch):
    client = _client(monkeypatch)
    response = client.post("/ask/stream", json={"question": "Why?"})
    assert response.status_code == 200
    assert "event: complete" in response.text
    assert _ledger_statuses() == ["completed"]


def test_stream_failure_releases(limits, monkeypatch):
    monkeypatch.setattr(routes, "delete_empty_conversation", Mock())
    client = _client(monkeypatch, prep=_Prep(RuntimeError("boom")))
    response = client.post("/ask/stream", json={"question": "Why?"})
    assert "event: error" in response.text
    assert _ledger_statuses() == ["failed"]


def test_stream_quota_rejected_before_streaming(limits, monkeypatch):
    _ledger("u", status="in_flight")
    client = _client(monkeypatch)
    response = client.post("/ask/stream", json={"question": "Why?"})
    assert response.status_code == 429
    assert response.json()["detail"] == routes.QUOTA_MESSAGES["concurrency"]


def test_finish_request_releases_unfinished_slot(limits):
    rid = str(uuid4())
    quotas.acquire_ask_slot("u", "free", rid)
    routes._finish_request(rid, "failed")
    assert _status(rid) == "failed"


def test_usage_endpoint(limits, monkeypatch):
    _ledger("u", tokens=300)
    client = _client(monkeypatch, plan="pro")
    body = client.get("/usage").json()
    assert body["plan"] == "pro"
    assert body["questions_used"] == 1
    assert body["questions_limit"] == 5
    assert body["tokens_used"] == 300
    assert body["max_concurrent"] == 2
    assert body["window_resets_at"] > time.time()


@pytest.fixture
def admin_client(monkeypatch):
    monkeypatch.setenv("DOUBTDESK_ADMIN_KEY", "test-secret-key")
    app = FastAPI()
    app.include_router(admin_router)
    return TestClient(app)


def test_admin_sets_plan(admin_client):
    user_id = _insert_user()
    headers = {"X-Admin-Key": "test-secret-key"}
    response = admin_client.patch(f"/admin/users/{user_id}/plan", json={"plan": "unlimited"}, headers=headers)
    assert response.status_code == 200
    assert response.json() == {"user_id": user_id, "plan": "unlimited"}
    assert admin_client.patch(f"/admin/users/{uuid4()}/plan", json={"plan": "pro"}, headers=headers).status_code == 404
    assert admin_client.patch(f"/admin/users/{user_id}/plan", json={"plan": "gold"}, headers=headers).status_code == 422
    assert admin_client.patch(f"/admin/users/{user_id}/plan", json={"plan": "pro"}).status_code == 401

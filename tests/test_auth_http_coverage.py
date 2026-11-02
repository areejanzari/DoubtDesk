import smtplib
import sqlite3
import time
from contextlib import closing

import pytest
from argon2 import PasswordHasher
from argon2.low_level import Type
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from doubtdesk.auth import dependencies, routes, security, store


@pytest.fixture
def auth_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("AUTH_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setattr(
        security,
        "password_hasher",
        PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1, type=Type.ID),
    )
    store.initialize_auth_db()
    sent_verification = []
    sent_reset = []

    def record_verification(to_email, token):
        sent_verification.append((to_email, token))

    def record_reset(to_email, token):
        sent_reset.append((to_email, token))

    monkeypatch.setattr(routes, "send_verification_email", record_verification)
    monkeypatch.setattr(routes, "send_password_reset_email", record_reset)

    app = FastAPI()
    app.include_router(routes.router)

    @app.get("/test/protected")
    def read_protected(user=Depends(dependencies.require_current_user)):
        return {"user_id": user["user_id"]}

    @app.post("/test/protected")
    def write_protected(user=Depends(dependencies.require_current_user)):
        return {"user_id": user["user_id"]}

    with TestClient(app) as client:
        yield client, sent_verification, sent_reset


def signup(client, username="student_a", email="student_a@example.com"):
    return client.post(
        "/auth/signup",
        json={"username": username, "email": email, "password": "original-password"},
    )


def verified_user(client, emails):
    registered = signup(client)
    assert registered.status_code == 201, registered.text
    assert emails
    verified = client.post("/auth/verify-email", json={"token": emails[-1][1]})
    assert verified.status_code == 200, verified.text
    return registered.json()


def login(client, identifier="student_a", password="original-password"):
    return client.post("/auth/login", json={"identifier": identifier, "password": password})


def sql(sql_text, args=()):
    with closing(sqlite3.connect(store.DB_PATH)) as conn, conn:
        return conn.execute(sql_text, args).fetchall()


def test_signup_creates_unverified_account_and_sets_no_session(auth_client):
    client, verifications, _ = auth_client
    response = signup(client)
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "student_a"
    assert data["email_verified"] is False
    assert "password" not in data
    assert verifications[0][0] == "student_a@example.com"
    assert "doubtdesk_session" not in client.cookies
    assert sql("SELECT verified_at FROM users WHERE user_id = ?", (data["user_id"],)) == [(None,)]


def test_signup_duplicate_username_and_email_rejected(auth_client):
    client, verifications, _ = auth_client
    assert signup(client).status_code == 201
    duplicate_username = signup(client, username="STUDENT_A", email="other@example.com")
    duplicate_email = signup(client, username="another", email="STUDENT_A@example.com")
    assert duplicate_username.status_code == 409
    assert duplicate_email.status_code == 409
    assert len(verifications) == 1


@pytest.mark.parametrize("payload", [
    {"username": "x", "email": "valid@example.com", "password": "good-password"},
    {"username": "good_name", "email": "invalid-email", "password": "good-password"},
    {"username": "good_name", "email": "valid@example.com", "password": "short"},
])
def test_signup_validation_rejects_bad_payload(auth_client, payload):
    client, sent, _ = auth_client
    assert client.post("/auth/signup", json=payload).status_code == 422
    assert not sent


def test_signup_email_delivery_failure_is_explicit(auth_client, monkeypatch):
    client, _, _ = auth_client
    def mail_failure(**kwargs):
        raise smtplib.SMTPException("smtp unavailable")
    monkeypatch.setattr(routes, "send_verification_email", mail_failure)
    response = signup(client)
    assert response.status_code == 503
    assert "verification email" in response.json()["detail"].lower()
    assert sql("SELECT COUNT(*) FROM users")[0] == (1,)


def test_verify_email_invalid_expired_and_replay(auth_client):
    client, sent, _ = auth_client
    assert signup(client).status_code == 201
    token = sent[-1][1]
    assert client.post("/auth/verify-email", json={"token": "nonsense"}).status_code == 400
    assert client.post("/auth/verify-email", json={"token": token}).status_code == 200
    assert client.post("/auth/verify-email", json={"token": token}).status_code == 400
    assert signup(client, "other_user", "other@example.com").status_code == 201
    expired = sent[-1][1]
    sql("UPDATE action_tokens SET expires_at = 0 WHERE token_hash = ?", (security.hash_token(expired),))
    assert client.post("/auth/verify-email", json={"token": expired}).status_code == 400


def test_resend_verification_never_exposes_account_existence(auth_client):
    client, sent, _ = auth_client
    response = client.post("/auth/resend-verification", json={"email": "absent@example.com"})
    assert response.status_code == 200
    generic = response.json()
    assert len(sent) == 0
    assert signup(client).status_code == 201
    initial = sent[-1][1]
    assert client.post("/auth/resend-verification", json={"email": "student_a@example.com"}).json() == generic
    assert len(sent) == 1
    sql("UPDATE action_tokens SET created_at = created_at - 61")
    assert client.post("/auth/resend-verification", json={"email": "student_a@example.com"}).json() == generic
    assert len(sent) == 2
    assert sent[-1][1] != initial
    assert client.post("/auth/verify-email", json={"token": sent[-1][1]}).status_code == 200
    assert client.post("/auth/resend-verification", json={"email": "student_a@example.com"}).json() == generic
    assert len(sent) == 2


def test_resend_verification_smtp_failure_is_generic(auth_client, monkeypatch):
    client, sent, _ = auth_client
    assert signup(client).status_code == 201
    sql("UPDATE action_tokens SET created_at = created_at - 61")
    monkeypatch.setattr(routes, "send_verification_email", lambda **kwargs: (_ for _ in ()).throw(OSError("smtp down")))
    response = client.post("/auth/resend-verification", json={"email": "student_a@example.com"})
    assert response.status_code == 200
    assert "unverified account" in response.json()["message"]
    assert len(sent) == 1


def test_login_requires_verified_account_and_valid_password(auth_client):
    client, sent, _ = auth_client
    assert signup(client).status_code == 201
    assert login(client).status_code == 401
    assert client.post("/auth/verify-email", json={"token": sent[-1][1]}).status_code == 200
    assert login(client, password="incorrect").status_code == 401
    assert login(client, identifier="unknown@example.com").status_code == 401
    ok = login(client, identifier="student_a@example.com")
    assert ok.status_code == 200
    assert ok.json()["csrf_token"]
    assert ok.json()["username"] == "student_a"
    assert "doubtdesk_session" in client.cookies
    assert "httponly" in ok.headers["set-cookie"].lower()
    assert "samesite=lax" in ok.headers["set-cookie"].lower()
    assert "max-age=14400" in ok.headers["set-cookie"].lower()


def test_login_rate_limit_returns_429_and_aliases_share_failure_count(auth_client):
    client, sent, _ = auth_client
    verified_user(client, sent)
    for i in range(5):
        identifier = "student_a" if i % 2 == 0 else "student_a@example.com"
        assert login(client, identifier=identifier, password="incorrect").status_code == 401
    assert login(client).status_code == 429
    assert login(client, identifier="student_a@example.com").status_code == 429


def test_session_restoration_cookie_csrf_and_logout(auth_client):
    client, sent, _ = auth_client
    user = verified_user(client, sent)
    assert client.get("/auth/me").status_code == 401
    assert client.post("/test/protected").status_code == 401
    response = login(client)
    assert response.status_code == 200
    csrf = response.json()["csrf_token"]
    me = client.get("/auth/me")
    assert me.status_code == 200
    assert me.json()["user_id"] == user["user_id"]
    assert me.json()["csrf_token"] == csrf
    assert client.get("/test/protected").status_code == 200
    assert client.post("/test/protected").status_code == 403
    assert client.post("/test/protected", headers={"X-CSRF-Token": "incorrect"}).status_code == 403
    assert client.post("/test/protected", headers={"X-CSRF-Token": csrf}).status_code == 200
    assert client.post("/auth/logout").status_code == 403
    out = client.post("/auth/logout", headers={"X-CSRF-Token": csrf})
    assert out.status_code == 200
    assert "doubtdesk_session" not in client.cookies
    assert client.get("/auth/me").status_code == 401


def test_expired_or_invalid_session_is_rejected(auth_client):
    client, sent, _ = auth_client
    verified_user(client, sent)
    assert login(client).status_code == 200
    sql("UPDATE sessions SET expires_at = 0")
    response = client.get("/auth/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired session."
    client.cookies.set("doubtdesk_session", "forged-token")
    assert client.get("/auth/me").status_code == 401


def test_origin_checked_for_writes_but_not_reads(auth_client):
    client, sent, _ = auth_client
    verified_user(client, sent)
    csrf = login(client).json()["csrf_token"]
    assert client.get("/test/protected", headers={"Origin": "https://attacker.example"}).status_code == 200
    assert client.post("/test/protected", headers={"Origin": "https://attacker.example", "X-CSRF-Token": csrf}).status_code == 403
    assert client.post("/test/protected", headers={"Origin": "http://localhost:5173", "X-CSRF-Token": csrf}).status_code == 200
    assert client.post("/test/protected", headers={"X-CSRF-Token": csrf}).status_code == 200


def test_allowed_origins_configuration_and_production_requirement(monkeypatch):
    monkeypatch.setenv("AUTH_ALLOWED_ORIGINS", " https://example.test/, https://other.test/ ")
    assert dependencies.allowed_origins() == {"https://example.test", "https://other.test"}
    monkeypatch.delenv("AUTH_ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("APP_ENV", "production")
    with pytest.raises(RuntimeError, match="AUTH_ALLOWED_ORIGINS"):
        dependencies.allowed_origins()
    monkeypatch.setenv("APP_ENV", "development")
    assert "http://localhost:5173" in dependencies.allowed_origins()


def test_forgot_password_generic_response_and_mail_failure(auth_client, monkeypatch):
    client, sent, resets = auth_client
    assert signup(client).status_code == 201
    unknown = client.post("/auth/forgot-password", json={"email": "unknown@example.com"})
    unverified = client.post("/auth/forgot-password", json={"email": "student_a@example.com"})
    assert unknown.status_code == unverified.status_code == 200
    assert unknown.json() == unverified.json()
    assert not resets
    assert client.post("/auth/verify-email", json={"token": sent[-1][1]}).status_code == 200
    valid = client.post("/auth/forgot-password", json={"email": "student_a@example.com"})
    assert valid.json() == unknown.json()
    assert len(resets) == 1
    assert client.post("/auth/forgot-password", json={"email": "student_a@example.com"}).json() == unknown.json()
    assert len(resets) == 1
    sql("UPDATE action_tokens SET created_at = created_at - 61 WHERE purpose = 'reset'")
    monkeypatch.setattr(routes, "send_password_reset_email", lambda **kwargs: (_ for _ in ()).throw(smtplib.SMTPException("down")))
    assert client.post("/auth/forgot-password", json={"email": "student_a@example.com"}).json() == unknown.json()


def test_reset_password_revokes_all_sessions_and_rejects_reuse(auth_client):
    client, sent, resets = auth_client
    verified_user(client, sent)
    first_login = login(client)
    assert first_login.status_code == 200
    old_cookie = client.cookies.get("doubtdesk_session")
    response = client.post("/auth/forgot-password", json={"email": "student_a@example.com"})
    assert response.status_code == 200
    token = resets[-1][1]
    assert client.post("/auth/reset-password", json={"token": "invalid", "new_password": "new-password"}).status_code == 400
    assert client.post("/auth/reset-password", json={"token": token, "new_password": "short"}).status_code == 422
    changed = client.post("/auth/reset-password", json={"token": token, "new_password": "new-password"})
    assert changed.status_code == 200
    assert client.post("/auth/reset-password", json={"token": token, "new_password": "new-password"}).status_code == 400
    client.cookies.set("doubtdesk_session", old_cookie)
    assert client.get("/auth/me").status_code == 401
    assert login(client).status_code == 401
    assert login(client, password="new-password").status_code == 200


def test_production_login_sets_secure_cookie(auth_client, monkeypatch):
    client, sent, _ = auth_client
    verified_user(client, sent)
    monkeypatch.setenv("APP_ENV", "production")
    response = login(client)
    assert response.status_code == 200
    assert "secure" in response.headers["set-cookie"].lower()

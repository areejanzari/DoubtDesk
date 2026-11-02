import hashlib
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from uuid import uuid4

import pytest
from argon2 import PasswordHasher
from argon2.low_level import Type

from doubtdesk.auth import security, service, store
from doubtdesk.auth.schemas import SignupRequest


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(
        security,
        "password_hasher",
        PasswordHasher(time_cost=1, memory_cost=1024, parallelism=1, type=Type.ID),
    )
    path = tmp_path / "auth.db"
    store.initialize_auth_db(path)
    return path


def insert_user(db_path, username="Student_A", email="student_a@example.com", verified=True):
    user_id = str(uuid4())
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            """
            INSERT INTO users (user_id, username, email, password_hash, verified_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                username,
                email,
                security.hash_password("correct-password"),
                int(time.time()) if verified else None,
                int(time.time()),
            ),
        )
    return user_id


def execute(db_path, sql, params=()):
    with closing(sqlite3.connect(db_path)) as conn, conn:
        return conn.execute(sql, params).fetchall()


def test_auth_schema_reinitializes_and_enforces_constraints(auth_db):
    store.initialize_auth_db(auth_db)
    user = insert_user(auth_db)
    with closing(sqlite3.connect(auth_db)) as conn, conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), "student_a", "different@example.com", "x", None, 1),
            )
    with closing(sqlite3.connect(auth_db)) as conn, conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO users VALUES (?, ?, ?, ?, ?, ?)",
                (str(uuid4()), "different", "STUDENT_A@example.com", "x", None, 1),
            )
    with closing(sqlite3.connect(auth_db)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                ("orphan", "missing-user", "csrf", 100, 1),
            )
    assert execute(auth_db, "SELECT user_id FROM users") == [(user,)]


def test_password_hashes_tokens_and_invalid_hashes(auth_db):
    hashed = security.hash_password("a secret")
    assert hashed != "a secret"
    assert security.verify_password("a secret", hashed)
    assert not security.verify_password("wrong", hashed)
    assert not security.verify_password("wrong", "not-an-argon2-hash")
    first = security.generate_token()
    second = security.generate_token()
    assert first != second
    assert len(first) >= 40
    assert security.hash_token(first) == hashlib.sha256(first.encode()).hexdigest()
    assert security.hash_token(first) != first


def test_register_and_authenticate_verified_and_unverified(auth_db):
    payload = SignupRequest(
        username="  new_student  ",
        email="registered@example.com",
        password="strong-password",
    )
    registered = service.register_user(payload, db_path=auth_db)
    assert registered["username"] == "new_student"
    assert registered["email_verified"] is False
    assert service.authenticate_user("new_student", "strong-password", auth_db) is None
    assert service.authenticate_user("missing", "strong-password", auth_db) is None
    assert service.authenticate_user("new_student", "wrong-password", auth_db) is None
    with closing(sqlite3.connect(auth_db)) as conn, conn:
        conn.execute("UPDATE users SET verified_at = ? WHERE user_id = ?", (int(time.time()), registered["user_id"]))
    user = service.authenticate_user(" registered@example.com ", "strong-password", auth_db)
    assert user["user_id"] == registered["user_id"]
    assert service.authenticate_user("new_student", "strong-password", auth_db) == user
    with pytest.raises(ValueError, match="already registered"):
        service.register_user(SignupRequest(username="NEW_STUDENT", email="newer@example.com", password="strong-password"), auth_db)
    with pytest.raises(ValueError, match="already registered"):
        service.register_user(SignupRequest(username="another", email="REGISTERED@example.com", password="strong-password"), auth_db)


@pytest.mark.parametrize(
    "username,password",
    [("x", "long-enough"), ("name with spaces", "long-enough"), ("valid_user", "short")],
)
def test_signup_validation_rejects_invalid_values(username, password):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SignupRequest(username=username, email="valid@example.com", password=password)


def test_verification_token_single_use_replacement_and_restrictions(auth_db):
    user_id = insert_user(auth_db, verified=False)
    with pytest.raises(ValueError, match="does not exist"):
        store.create_verification_token("missing-user", auth_db)
    token_old = store.create_verification_token(user_id, auth_db)
    token_new = store.create_verification_token(user_id, auth_db)
    assert not store.verify_email_token(token_old, auth_db)
    assert store.verify_email_token(token_new, auth_db)
    assert not store.verify_email_token(token_new, auth_db)
    with pytest.raises(ValueError, match="already verified"):
        store.create_verification_token(user_id, auth_db)


def test_expired_verification_token_is_rejected_and_removed(auth_db):
    user_id = insert_user(auth_db, verified=False)
    token = store.create_verification_token(user_id, auth_db)
    execute(auth_db, "UPDATE action_tokens SET expires_at = 0")
    assert not store.verify_email_token(token, auth_db)
    assert execute(auth_db, "SELECT COUNT(*) FROM action_tokens")[0][0] == 0
    assert execute(auth_db, "SELECT verified_at FROM users WHERE user_id = ?", (user_id,)) == [(None,)]


def test_resend_verification_unknown_verified_cooldown_and_rotation(auth_db):
    user_id = insert_user(auth_db, verified=False)
    insert_user(auth_db, username="Verified", email="verified@example.com")
    assert store.resend_verification_token("absent@example.com", auth_db) is None
    assert store.resend_verification_token("verified@example.com", auth_db) is None
    first = store.resend_verification_token("student_a@example.com", auth_db)
    assert first
    assert store.resend_verification_token("student_a@example.com", auth_db) is None
    execute(auth_db, "UPDATE action_tokens SET created_at = created_at - 61 WHERE user_id = ?", (user_id,))
    second = store.resend_verification_token("student_a@example.com", auth_db)
    assert second and second != first
    assert not store.verify_email_token(first, auth_db)
    assert store.verify_email_token(second, auth_db)


def test_session_rejects_missing_and_unverified_users(auth_db):
    unverified = insert_user(auth_db, verified=False)
    with pytest.raises(ValueError, match="not verified"):
        store.create_session(unverified, auth_db)
    with pytest.raises(ValueError, match="not verified"):
        store.create_session("unknown", auth_db)
    assert store.get_session("bogus", auth_db) is None


def test_session_is_hashed_expires_and_revoke_is_idempotent(auth_db):
    user_id = insert_user(auth_db)
    token, csrf = store.create_session(user_id, auth_db)
    assert csrf and csrf != token
    saved = store.get_session(token, auth_db)
    assert saved["user_id"] == user_id
    assert saved["csrf_token"] == csrf
    assert saved["email"] == "student_a@example.com"
    row = execute(auth_db, "SELECT token_hash, expires_at, created_at FROM sessions")[0]
    assert row[0] == security.hash_token(token)
    assert row[0] != token
    assert row[1] - row[2] == 14400
    execute(auth_db, "UPDATE sessions SET expires_at = 0")
    assert store.get_session(token, auth_db) is None
    new_token, _ = store.create_session(user_id, auth_db)
    store.revoke_session(new_token, auth_db)
    store.revoke_session(new_token, auth_db)
    assert store.get_session(new_token, auth_db) is None


def test_password_reset_requires_verified_user_cooldown_rotation_and_replay(auth_db):
    user_id = insert_user(auth_db)
    insert_user(auth_db, username="Unverified", email="unverified@example.com", verified=False)
    assert store.create_password_reset_token("missing@example.com", auth_db) is None
    assert store.create_password_reset_token("unverified@example.com", auth_db) is None
    first = store.create_password_reset_token("student_a@example.com", auth_db)
    assert first
    assert store.create_password_reset_token("student_a@example.com", auth_db) is None
    execute(auth_db, "UPDATE action_tokens SET created_at = created_at - 61 WHERE user_id = ?", (user_id,))
    second = store.create_password_reset_token("student_a@example.com", auth_db)
    assert second and second != first
    assert not store.reset_password(first, "new-password", auth_db)
    token1, _ = store.create_session(user_id, auth_db)
    token2, _ = store.create_session(user_id, auth_db)
    verify_token_row = execute(auth_db, "SELECT COUNT(*) FROM action_tokens WHERE purpose = 'reset'")[0][0]
    assert verify_token_row == 1
    assert store.reset_password(second, "new-password", auth_db)
    assert not store.reset_password(second, "another-password", auth_db)
    assert not store.get_session(token1, auth_db)
    assert not store.get_session(token2, auth_db)
    assert execute(auth_db, "SELECT COUNT(*) FROM action_tokens WHERE user_id = ?", (user_id,))[0][0] == 0
    assert service.authenticate_user("student_a@example.com", "correct-password", auth_db) is None
    assert service.authenticate_user("Student_A", "new-password", auth_db)


def test_expired_password_reset_does_not_change_password(auth_db):
    user_id = insert_user(auth_db)
    token = store.create_password_reset_token("student_a@example.com", auth_db)
    execute(auth_db, "UPDATE action_tokens SET expires_at = 0")
    assert not store.reset_password(token, "new-password", auth_db)
    assert service.authenticate_user("Student_A", "correct-password", auth_db)
    assert execute(auth_db, "SELECT COUNT(*) FROM action_tokens")[0][0] == 0


def test_identifier_hash_normalizes_aliases_and_unknowns(auth_db):
    user_id = insert_user(auth_db)
    with closing(sqlite3.connect(auth_db)) as conn:
        by_name = store.get_login_identifier_hash(conn, " STUDENT_A ")
        by_email = store.get_login_identifier_hash(conn, "student_a@example.com")
        unknown_1 = store.get_login_identifier_hash(conn, "  Nobody  ")
        unknown_2 = store.get_login_identifier_hash(conn, "nobody")
    assert by_name == by_email
    assert by_name == hashlib.sha256(f"user:{user_id}".encode()).hexdigest()
    assert unknown_1 == unknown_2
    assert unknown_1 != by_name


def test_login_rate_limited_by_account_shared_across_aliases(auth_db):
    insert_user(auth_db)
    for _ in range(5):
        assert store.record_failed_login("Student_A", "10.0.0.1", auth_db)
    assert store.is_login_rate_limited("student_a@example.com", "10.0.0.2", auth_db)
    assert not store.record_failed_login("student_a@example.com", "10.0.0.3", auth_db)
    assert not store.is_login_rate_limited("unknown@example.com", "10.0.0.3", auth_db)


def test_login_rate_limited_by_ip_and_old_records_ignored(auth_db):
    for i in range(20):
        assert store.record_failed_login(f"different_{i}", "198.51.100.4", auth_db)
    assert store.is_login_rate_limited("a-different-account", "198.51.100.4", auth_db)
    assert not store.record_failed_login("other", "198.51.100.4", auth_db)
    execute(auth_db, "UPDATE login_failures SET attempted_at = attempted_at - 901")
    assert not store.is_login_rate_limited("other", "198.51.100.4", auth_db)
    assert store.record_failed_login("other", "198.51.100.4", auth_db)


def test_concurrent_failures_limit_atomic_writes(auth_db):
    insert_user(auth_db)
    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(lambda _: store.record_failed_login("Student_A", "192.0.2.1", auth_db), range(9)))
    assert outcomes.count(True) == 5
    assert outcomes.count(False) == 4
    assert execute(auth_db, "SELECT COUNT(*) FROM login_failures")[0][0] == 5


def test_cleanup_removes_only_expired_records(auth_db):
    user_id = insert_user(auth_db)
    old_token, _ = store.create_session(user_id, auth_db)
    new_token, _ = store.create_session(user_id, auth_db)
    reset = store.create_password_reset_token("student_a@example.com", auth_db)
    assert reset
    store.record_failed_login("Student_A", "192.0.2.1", auth_db)
    store.record_failed_login("Student_A", "192.0.2.1", auth_db)
    execute(auth_db, "UPDATE sessions SET expires_at = 0 WHERE token_hash = ?", (security.hash_token(old_token),))
    execute(auth_db, "UPDATE action_tokens SET expires_at = 0")
    execute(auth_db, "UPDATE login_failures SET attempted_at = 0 WHERE rowid = (SELECT MIN(rowid) FROM login_failures)")
    result = store.cleanup_expired_auth_records(auth_db)
    assert result == {"sessions_deleted": 1, "tokens_deleted": 1, "failures_deleted": 1}
    assert store.get_session(new_token, auth_db)
    assert execute(auth_db, "SELECT COUNT(*) FROM users")[0][0] == 1
    assert store.cleanup_expired_auth_records(auth_db) == {"sessions_deleted": 0, "tokens_deleted": 0, "failures_deleted": 0}


def test_signup_username_validator_preserves_type_errors():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        SignupRequest.model_validate({
            "username": 123,
            "email": "valid@example.com",
            "password": "long-enough",
        })

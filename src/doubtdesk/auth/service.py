import sqlite3
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4
from doubtdesk.auth.schemas import SignupRequest
from doubtdesk.auth.security import hash_password, verify_password
from doubtdesk.auth.store import DB_PATH


def register_user(
    payload: SignupRequest,
    db_path: Path = DB_PATH,
) -> dict:
    user_id = str(uuid4())
    password_hash = hash_password(payload.password)
    created_at = int(time.time())

    try:
        with closing(sqlite3.connect(db_path)) as conn, conn:
            conn.execute(
                """
                INSERT INTO users (
                    user_id,
                    username,
                    email,
                    password_hash,
                    verified_at,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    payload.username,
                    str(payload.email),
                    password_hash,
                    None,
                    created_at,
                ),
            )

    except sqlite3.IntegrityError as exc:
        raise ValueError(
            "Username or email already registered."
        ) from exc

    return {
        "user_id": user_id,
        "username": payload.username,
        "email": str(payload.email),
        "email_verified": False,
    }


def authenticate_user(
    identifier: str,
    password: str,
    db_path: Path = DB_PATH,
) -> dict | None:
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        user = conn.execute(
            """
            SELECT user_id, username, email, password_hash, verified_at
            FROM users
            WHERE username = ? OR email = ?
            """,
            (identifier.strip(), identifier.strip()),
        ).fetchone()

    if user is None:
        return None

    if not verify_password(password, user["password_hash"]):
        return None

    if user["verified_at"] is None:
        return None

    return {
        "user_id": user["user_id"],
        "username": user["username"],
        "email": user["email"],
    }

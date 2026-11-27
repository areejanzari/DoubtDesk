import time
from uuid import uuid4
import psycopg
from psycopg.rows import dict_row
from doubtdesk.auth.schemas import SignupRequest
from doubtdesk.auth.security import hash_password, verify_password
from doubtdesk.db import connection

DUMMY_PASSWORD_HASH = '$argon2id$v=19$m=65536,t=3,p=4$LPSC2j7Xy54ovZJndwnF6A$Skiafz8somdqy8NCud/CcIyf2pQumj1QTQVitQtjE88'


def register_user(
    payload: SignupRequest,
) -> dict:
    user_id = str(uuid4())
    password_hash = hash_password(payload.password)
    created_at = int(time.time())

    try:
        with connection() as conn:
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
                VALUES (%s, %s, %s, %s, %s, %s)
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

    except psycopg.errors.UniqueViolation as exc:
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
) -> dict | None:
    with connection() as conn:
        user = conn.cursor(row_factory=dict_row).execute(
            """
            SELECT user_id, username::text AS username, email::text AS email, password_hash, verified_at
            FROM users
            WHERE username = %s OR email = %s
            """,
            (identifier.strip(), identifier.strip()),
        ).fetchone()

    if user is None:
        verify_password(password, DUMMY_PASSWORD_HASH)
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

import hashlib
import time
from psycopg import Connection
from psycopg.rows import dict_row
from doubtdesk.auth.security import generate_token, hash_token, hash_password
from doubtdesk.db import connection


def _replace_action_token(conn, user_id, purpose, created_at):
    token = generate_token()
    conn.execute("DELETE FROM action_tokens WHERE user_id = %s AND purpose = %s", (user_id, purpose))
    conn.execute("""INSERT INTO action_tokens (token_hash, user_id, purpose, expires_at, created_at)
                     VALUES (%s, %s, %s, %s, %s)""", (hash_token(token), user_id, purpose, created_at + 600, created_at))
    return token


def create_verification_token(
    user_id: str,
) -> str:
    created_at = int(time.time())

    with connection() as conn:
        user = conn.execute(
            "SELECT verified_at FROM users WHERE user_id = %s FOR UPDATE",
            (user_id,),
        ).fetchone()

        if user is None:
            raise ValueError("User does not exist.")

        if user[0] is not None:
            raise ValueError("Email is already verified.")

        token = _replace_action_token(conn, user_id, "verify", created_at)

    return token


def verify_email_token(
    token: str,
) -> bool:
    token_hash = hash_token(token)
    current_time = int(time.time())

    with connection() as conn:
        row = conn.execute(
            """
            SELECT user_id, expires_at
            FROM action_tokens
            WHERE token_hash = %s AND purpose = 'verify'
            FOR UPDATE
            """,
            (token_hash,),
        ).fetchone()

        if row is None:
            return False

        user_id, expires_at = row

        if expires_at <= current_time:
            conn.execute(
                "DELETE FROM action_tokens WHERE token_hash = %s",
                (token_hash,),
            )
            return False

        conn.execute(
            """
            UPDATE users
            SET verified_at = %s
            WHERE user_id = %s AND verified_at IS NULL
            """,
            (current_time, user_id),
        )

        conn.execute(
            "DELETE FROM action_tokens WHERE token_hash = %s",
            (token_hash,),
        )

    return True


def resend_verification_token(
    email: str,
) -> str | None:
    current_time = int(time.time())

    with connection() as conn:
        user = conn.execute(
            """
            SELECT user_id, verified_at
            FROM users
            WHERE email = %s
            FOR UPDATE
            """,
            (email,),
        ).fetchone()

        if user is None:
            return None

        user_id, verified_at = user

        if verified_at is not None:
            return None

        latest_token = conn.execute(
            """
            SELECT created_at
            FROM action_tokens
            WHERE user_id = %s AND purpose = 'verify'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        if latest_token is not None:
            if current_time - latest_token[0] < 60:
                return None

        token = generate_token()

        conn.execute(
            """
            DELETE FROM action_tokens
            WHERE user_id = %s AND purpose = 'verify'
            """,
            (user_id,),
        )

        conn.execute(
            """
            INSERT INTO action_tokens (
                token_hash,
                user_id,
                purpose,
                expires_at,
                created_at
            )
            VALUES (%s, %s, 'verify', %s, %s)
            """,
            (
                hash_token(token),
                user_id,
                current_time + 600,
                current_time,
            ),
        )

    return token


def create_session(
    user_id: str,
) -> tuple[str, str]:
    session_token = generate_token()
    csrf_token = generate_token()

    created_at = int(time.time())
    expires_at = created_at + (4 * 60 * 60)

    with connection() as conn:
        user = conn.execute(
            "SELECT verified_at FROM users WHERE user_id = %s",
            (user_id,),
        ).fetchone()

        if user is None or user[0] is None:
            raise ValueError("User is not verified.")

        conn.execute(
            """
            INSERT INTO sessions (
                token_hash,
                user_id,
                csrf_token,
                expires_at,
                created_at
            )
            VALUES (%s, %s, %s, %s, %s)
            """,
            (
                hash_token(session_token),
                user_id,
                csrf_token,
                expires_at,
                created_at,
            ),
        )

    return session_token, csrf_token



def get_session(
    session_token: str,
) -> dict | None:
    token_hash = hash_token(session_token)
    current_time = int(time.time())

    with connection() as conn:
        session = conn.cursor(row_factory=dict_row).execute(
            """
            SELECT
                s.user_id,
                s.csrf_token,
                s.expires_at,
                u.username::text AS username,
                u.email::text AS email,
                u.plan
            FROM sessions AS s
            JOIN users AS u ON s.user_id = u.user_id
            WHERE s.token_hash = %s
              AND s.expires_at > %s
              AND u.verified_at IS NOT NULL
            """,
            (token_hash, current_time),
        ).fetchone()

    return session


def revoke_session(
    session_token: str,
) -> None:
    token_hash = hash_token(session_token)

    with connection() as conn:
        conn.execute(
            "DELETE FROM sessions WHERE token_hash = %s",
            (token_hash,),
        )


def create_password_reset_token(
    email: str,
) -> str | None:
    current_time = int(time.time())

    with connection() as conn:
        user = conn.execute(
            """
            SELECT user_id, verified_at
            FROM users
            WHERE email = %s
            FOR UPDATE
            """,
            (email,),
        ).fetchone()

        if user is None or user[1] is None:
            return None

        user_id = user[0]

        latest_token = conn.execute(
            """
            SELECT created_at
            FROM action_tokens
            WHERE user_id = %s AND purpose = 'reset'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (user_id,),
        ).fetchone()

        if latest_token is not None:
            if current_time - latest_token[0] < 60:
                return None

        token = _replace_action_token(conn, user_id, "reset", current_time)

    return token


def reset_password(
    token: str,
    new_password: str,
) -> bool:
    token_hash = hash_token(token)
    current_time = int(time.time())

    with connection() as conn:
        row = conn.execute(
            """
            SELECT user_id, expires_at
            FROM action_tokens
            WHERE token_hash = %s AND purpose = 'reset'
            FOR UPDATE
            """,
            (token_hash,),
        ).fetchone()

        if row is None:
            return False

        user_id, expires_at = row

        if expires_at <= current_time:
            conn.execute(
                "DELETE FROM action_tokens WHERE token_hash = %s",
                (token_hash,),
            )
            return False

        password_hash = hash_password(new_password)
        conn.execute(
            """
            UPDATE users
            SET password_hash = %s
            WHERE user_id = %s AND verified_at IS NOT NULL
            """,
            (password_hash, user_id),
        )

        conn.execute(
            "DELETE FROM sessions WHERE user_id = %s",
            (user_id,),
        )

        conn.execute(
            "DELETE FROM action_tokens WHERE user_id = %s",
            (user_id,),
        )

    return True



LOGIN_WINDOW = 15 * 60
MAX_IDENTIFIER_FAILURES = 5
MAX_IP_FAILURES = 20


def get_login_identifier_hash(
    conn: Connection,
    identifier: str,
) -> str:
    normalized = identifier.strip().casefold()

    row = conn.execute(
        """
        SELECT user_id
        FROM users
        WHERE username = %s OR email = %s
        """,
        (normalized, normalized),
    ).fetchone()

    if row is not None:
        value = f"user:{row[0]}"
    else:
        value = f"unknown:{normalized}"

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _login_counts(conn, identifier_hash, ip_address, cutoff):
    identifier_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM login_failures
        WHERE identifier_hash = %s
          AND attempted_at > %s
        """,
        (identifier_hash, cutoff),
    ).fetchone()[0]

    ip_count = conn.execute(
        """
        SELECT COUNT(*)
        FROM login_failures
        WHERE ip_address = %s
          AND attempted_at > %s
        """,
        (ip_address, cutoff),
    ).fetchone()[0]
    return identifier_count, ip_count


def is_login_rate_limited(
    identifier: str,
    ip_address: str,
) -> bool:
    cutoff = int(time.time()) - LOGIN_WINDOW

    with connection() as conn:
        identifier_hash = get_login_identifier_hash(
            conn, identifier
        )

        identifier_count, ip_count = _login_counts(conn, identifier_hash, ip_address, cutoff)

    return (
        identifier_count >= MAX_IDENTIFIER_FAILURES
        or ip_count >= MAX_IP_FAILURES
    )


def record_failed_login(
    identifier: str,
    ip_address: str,
) -> bool:
    now = int(time.time())
    cutoff = now - LOGIN_WINDOW

    with connection() as conn:
        conn.execute("LOCK TABLE login_failures IN SHARE ROW EXCLUSIVE MODE")

        identifier_hash = get_login_identifier_hash(
            conn, identifier
        )

        identifier_count, ip_count = _login_counts(conn, identifier_hash, ip_address, cutoff)

        if (
            identifier_count >= MAX_IDENTIFIER_FAILURES
            or ip_count >= MAX_IP_FAILURES
        ):
            return False

        conn.execute(
            """
            INSERT INTO login_failures (
                identifier_hash,
                ip_address,
                attempted_at
            )
            VALUES (%s, %s, %s)
            """,
            (identifier_hash, ip_address, now),
        )

    return True


def cleanup_expired_auth_records() -> dict[str, int]:
    now = int(time.time())
    login_cutoff = now - LOGIN_WINDOW

    with connection() as conn:
        sessions = conn.execute(
            "DELETE FROM sessions WHERE expires_at <= %s",
            (now,),
        ).rowcount

        tokens = conn.execute(
            "DELETE FROM action_tokens WHERE expires_at <= %s",
            (now,),
        ).rowcount

        conn.execute("DELETE FROM auth_attempts WHERE attempted_at <= %s", (login_cutoff,))
        failures = conn.execute(
            "DELETE FROM login_failures WHERE attempted_at <= %s",
            (login_cutoff,),
        ).rowcount

    return {
        "sessions_deleted": sessions,
        "tokens_deleted": tokens,
        "failures_deleted": failures,
    }


def consume_auth_attempt(action, ip_address, email):
    now = int(time.time())
    cutoff = now - LOGIN_WINDOW
    email_hash = hashlib.sha256(email.strip().casefold().encode("utf-8")).hexdigest()
    with connection() as conn:
        conn.execute("LOCK TABLE auth_attempts IN SHARE ROW EXCLUSIVE MODE")
        ip_count = conn.execute("SELECT COUNT(*) FROM auth_attempts WHERE action = %s AND ip_address = %s AND attempted_at > %s", (action, ip_address, cutoff)).fetchone()[0]
        email_count = conn.execute("SELECT COUNT(*) FROM auth_attempts WHERE action = %s AND email_hash = %s AND attempted_at > %s", (action, email_hash, cutoff)).fetchone()[0]
        if ip_count >= 20 or email_count >= 5:
            return False
        conn.execute("INSERT INTO auth_attempts (action, ip_address, email_hash, attempted_at) VALUES (%s, %s, %s, %s)", (action, ip_address, email_hash, now))
    return True

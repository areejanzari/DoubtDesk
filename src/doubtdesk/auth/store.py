import sqlite3
import hashlib
import time
from contextlib import closing
from pathlib import Path
from doubtdesk.auth.security import generate_token, hash_token, hash_password

DB_PATH = Path("data/auth.db")


def initialize_auth_db(db_path: Path = DB_PATH) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                verified_at INTEGER,
                created_at INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                csrf_token TEXT NOT NULL,
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sessions_user
            ON sessions(user_id);

            CREATE TABLE IF NOT EXISTS action_tokens (
                token_hash TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id),
                purpose TEXT NOT NULL CHECK (
                    purpose IN ('verify', 'reset')
                ),
                expires_at INTEGER NOT NULL,
                created_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_action_tokens_user
            ON action_tokens(user_id, purpose);

            CREATE TABLE IF NOT EXISTS login_failures (
                identifier_hash TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                attempted_at INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_login_failures_time
            ON login_failures(attempted_at);

            CREATE INDEX IF NOT EXISTS idx_login_failures_identifier
            ON login_failures(identifier_hash, attempted_at);

            CREATE INDEX IF NOT EXISTS idx_login_failures_ip
            ON login_failures(ip_address, attempted_at);
            """
        )


def create_verification_token(
    user_id: str,
    db_path: Path = DB_PATH,
) -> str:
    token = generate_token()
    token_hash = hash_token(token)

    created_at = int(time.time())
    expires_at = created_at + (10 * 60)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")

        user = conn.execute(
            "SELECT verified_at FROM users WHERE user_id = ?",
            (user_id,),
        ).fetchone()

        if user is None:
            raise ValueError("User does not exist.")

        if user[0] is not None:
            raise ValueError("Email is already verified.")

        conn.execute(
            """
            DELETE FROM action_tokens
            WHERE user_id = ? AND purpose = 'verify'
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
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                token_hash,
                user_id,
                "verify",
                expires_at,
                created_at,
            ),
        )

    return token


def verify_email_token(
    token: str,
    db_path: Path = DB_PATH,
) -> bool:
    token_hash = hash_token(token)
    current_time = int(time.time())

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            """
            SELECT user_id, expires_at
            FROM action_tokens
            WHERE token_hash = ? AND purpose = 'verify'
            """,
            (token_hash,),
        ).fetchone()

        if row is None:
            return False

        user_id, expires_at = row

        if expires_at <= current_time:
            conn.execute(
                "DELETE FROM action_tokens WHERE token_hash = ?",
                (token_hash,),
            )
            return False

        conn.execute(
            """
            UPDATE users
            SET verified_at = ?
            WHERE user_id = ? AND verified_at IS NULL
            """,
            (current_time, user_id),
        )

        conn.execute(
            "DELETE FROM action_tokens WHERE token_hash = ?",
            (token_hash,),
        )

    return True


def resend_verification_token(
    email: str,
    db_path: Path = DB_PATH,
) -> str | None:
    current_time = int(time.time())

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")

        user = conn.execute(
            """
            SELECT user_id, verified_at
            FROM users
            WHERE email = ?
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
            WHERE user_id = ? AND purpose = 'verify'
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
            WHERE user_id = ? AND purpose = 'verify'
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
            VALUES (?, ?, 'verify', ?, ?)
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
    db_path: Path = DB_PATH,
) -> tuple[str, str]:
    session_token = generate_token()
    csrf_token = generate_token()

    created_at = int(time.time())
    expires_at = created_at + (4 * 60 * 60)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")

        user = conn.execute(
            "SELECT verified_at FROM users WHERE user_id = ?",
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
            VALUES (?, ?, ?, ?, ?)
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
    db_path: Path = DB_PATH,
) -> dict | None:
    token_hash = hash_token(session_token)
    current_time = int(time.time())

    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row

        session = conn.execute(
            """
            SELECT
                s.user_id,
                s.csrf_token,
                s.expires_at,
                u.username,
                u.email
            FROM sessions AS s
            JOIN users AS u ON s.user_id = u.user_id
            WHERE s.token_hash = ?
              AND s.expires_at > ?
              AND u.verified_at IS NOT NULL
            """,
            (token_hash, current_time),
        ).fetchone()

    if session is None:
        return None

    return dict(session)


def revoke_session(
    session_token: str,
    db_path: Path = DB_PATH,
) -> None:
    token_hash = hash_token(session_token)

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (token_hash,),
        )


def create_password_reset_token(
    email: str,
    db_path: Path = DB_PATH,
) -> str | None:
    current_time = int(time.time())

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")

        user = conn.execute(
            """
            SELECT user_id, verified_at
            FROM users
            WHERE email = ?
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
            WHERE user_id = ? AND purpose = 'reset'
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
            WHERE user_id = ? AND purpose = 'reset'
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
            VALUES (?, ?, 'reset', ?, ?)
            """,
            (
                hash_token(token),
                user_id,
                current_time + 600,
                current_time,
            ),
        )

    return token


def reset_password(
    token: str,
    new_password: str,
    db_path: Path = DB_PATH,
) -> bool:
    token_hash = hash_token(token)
    password_hash = hash_password(new_password)
    current_time = int(time.time())

    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("BEGIN IMMEDIATE")

        row = conn.execute(
            """
            SELECT user_id, expires_at
            FROM action_tokens
            WHERE token_hash = ? AND purpose = 'reset'
            """,
            (token_hash,),
        ).fetchone()

        if row is None:
            return False

        user_id, expires_at = row

        if expires_at <= current_time:
            conn.execute(
                "DELETE FROM action_tokens WHERE token_hash = ?",
                (token_hash,),
            )
            return False

        conn.execute(
            """
            UPDATE users
            SET password_hash = ?
            WHERE user_id = ? AND verified_at IS NOT NULL
            """,
            (password_hash, user_id),
        )

        conn.execute(
            "DELETE FROM sessions WHERE user_id = ?",
            (user_id,),
        )

        conn.execute(
            "DELETE FROM action_tokens WHERE user_id = ?",
            (user_id,),
        )

    return True



LOGIN_WINDOW = 15 * 60
MAX_IDENTIFIER_FAILURES = 5
MAX_IP_FAILURES = 20


def get_login_identifier_hash(
    conn: sqlite3.Connection,
    identifier: str,
) -> str:
    normalized = identifier.strip().casefold()

    row = conn.execute(
        """
        SELECT user_id
        FROM users
        WHERE username = ? OR email = ?
        """,
        (normalized, normalized),
    ).fetchone()

    if row is not None:
        value = f"user:{row[0]}"
    else:
        value = f"unknown:{normalized}"

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def is_login_rate_limited(
    identifier: str,
    ip_address: str,
    db_path: Path = DB_PATH,
) -> bool:
    cutoff = int(time.time()) - LOGIN_WINDOW

    with closing(sqlite3.connect(db_path)) as conn:
        identifier_hash = get_login_identifier_hash(
            conn, identifier
        )

        identifier_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM login_failures
            WHERE identifier_hash = ?
              AND attempted_at > ?
            """,
            (identifier_hash, cutoff),
        ).fetchone()[0]

        ip_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM login_failures
            WHERE ip_address = ?
              AND attempted_at > ?
            """,
            (ip_address, cutoff),
        ).fetchone()[0]

    return (
        identifier_count >= MAX_IDENTIFIER_FAILURES
        or ip_count >= MAX_IP_FAILURES
    )


def record_failed_login(
    identifier: str,
    ip_address: str,
    db_path: Path = DB_PATH,
) -> bool:
    now = int(time.time())
    cutoff = now - LOGIN_WINDOW

    with closing(sqlite3.connect(db_path, timeout=15)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")

        identifier_hash = get_login_identifier_hash(
            conn, identifier
        )

        identifier_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM login_failures
            WHERE identifier_hash = ?
              AND attempted_at > ?
            """,
            (identifier_hash, cutoff),
        ).fetchone()[0]

        ip_count = conn.execute(
            """
            SELECT COUNT(*)
            FROM login_failures
            WHERE ip_address = ?
              AND attempted_at > ?
            """,
            (ip_address, cutoff),
        ).fetchone()[0]

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
            VALUES (?, ?, ?)
            """,
            (identifier_hash, ip_address, now),
        )

    return True


def cleanup_expired_auth_records(
    db_path: Path = DB_PATH,
) -> dict[str, int]:
    now = int(time.time())
    login_cutoff = now - LOGIN_WINDOW

    with closing(sqlite3.connect(db_path)) as conn, conn:
        sessions = conn.execute(
            "DELETE FROM sessions WHERE expires_at <= ?",
            (now,),
        ).rowcount

        tokens = conn.execute(
            "DELETE FROM action_tokens WHERE expires_at <= ?",
            (now,),
        ).rowcount

        failures = conn.execute(
            "DELETE FROM login_failures WHERE attempted_at <= ?",
            (login_cutoff,),
        ).rowcount

    return {
        "sessions_deleted": sessions,
        "tokens_deleted": tokens,
        "failures_deleted": failures,
    }

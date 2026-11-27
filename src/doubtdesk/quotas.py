import time
from dataclasses import dataclass

from doubtdesk.config import runtime_settings
from doubtdesk.db import connection

WINDOW_SECONDS = 24 * 60 * 60
RETENTION_SECONDS = 2 * WINDOW_SECONDS
CONCURRENCY_RETRY_SECONDS = 5
PLAN_NAMES = ("free", "pro", "unlimited")


@dataclass(frozen=True)
class PlanLimits:
    daily_questions: int | None
    daily_tokens: int | None
    max_concurrent: int | None


class QuotaExceeded(Exception):
    def __init__(self, reason: str, retry_after: int):
        self.reason = reason
        self.retry_after = max(1, int(retry_after))
        super().__init__(reason)


def plan_limits(plan: str) -> PlanLimits:
    config = runtime_settings()
    if plan == "unlimited":
        return PlanLimits(None, None, None)
    if plan == "pro":
        return PlanLimits(config.pro_daily_questions, config.pro_daily_tokens, config.pro_max_concurrent)
    return PlanLimits(config.free_daily_questions, config.free_daily_tokens, config.free_max_concurrent)


def _window_usage(conn, user_id, now, stale_seconds):
    window_start = now - WINDOW_SECONDS
    questions, oldest = conn.execute(
        """
        SELECT COUNT(*), MIN(started_at)
        FROM ask_requests
        WHERE user_id = %s AND started_at > %s
        """,
        (user_id, window_start),
    ).fetchone()
    tokens = conn.execute(
        """
        SELECT COALESCE(SUM(u.total_tokens), 0)
        FROM ask_requests AS r
        JOIN ai_usage AS u ON u.request_id = r.request_id
        WHERE r.user_id = %s AND r.started_at > %s
        """,
        (user_id, window_start),
    ).fetchone()[0]
    in_flight = conn.execute(
        """
        SELECT COUNT(*)
        FROM ask_requests
        WHERE user_id = %s AND status = 'in_flight' AND started_at > %s
        """,
        (user_id, now - stale_seconds),
    ).fetchone()[0]
    return int(questions), int(tokens), int(in_flight), oldest


def acquire_ask_slot(user_id: str, plan: str, request_id: str) -> None:
    limits = plan_limits(plan)
    stale_seconds = runtime_settings().inflight_stale_seconds
    now = int(time.time())
    with connection() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"ask:{user_id}",))
        questions, tokens, in_flight, oldest = _window_usage(conn, user_id, now, stale_seconds)
        reset_in = (oldest + WINDOW_SECONDS - now) if oldest is not None else WINDOW_SECONDS
        if limits.max_concurrent is not None and in_flight >= limits.max_concurrent:
            raise QuotaExceeded("concurrency", CONCURRENCY_RETRY_SECONDS)
        if limits.daily_questions is not None and questions >= limits.daily_questions:
            raise QuotaExceeded("questions", reset_in)
        if limits.daily_tokens is not None and tokens >= limits.daily_tokens:
            raise QuotaExceeded("tokens", reset_in)
        conn.execute(
            """
            INSERT INTO ask_requests (request_id, user_id, status, started_at)
            VALUES (%s, %s, 'in_flight', %s)
            """,
            (request_id, user_id, now),
        )


def release_ask_slot(request_id: str, status: str) -> None:
    with connection() as conn:
        conn.execute(
            """
            UPDATE ask_requests
            SET status = %s, finished_at = %s
            WHERE request_id = %s AND status = 'in_flight'
            """,
            (status, int(time.time()), request_id),
        )


def get_user_usage(user_id: str, plan: str) -> dict:
    limits = plan_limits(plan)
    now = int(time.time())
    with connection() as conn:
        questions, tokens, in_flight, oldest = _window_usage(
            conn, user_id, now, runtime_settings().inflight_stale_seconds,
        )
    return {
        "plan": plan,
        "questions_used": questions,
        "questions_limit": limits.daily_questions,
        "tokens_used": tokens,
        "tokens_limit": limits.daily_tokens,
        "in_flight": in_flight,
        "max_concurrent": limits.max_concurrent,
        "window_resets_at": (oldest + WINDOW_SECONDS) if oldest is not None else None,
    }


def set_user_plan(user_id: str, plan: str) -> bool:
    with connection() as conn:
        updated = conn.execute(
            "UPDATE users SET plan = %s WHERE user_id = %s",
            (plan, user_id),
        ).rowcount
    return updated > 0


def cleanup_ask_requests() -> int:
    cutoff = int(time.time()) - RETENTION_SECONDS
    with connection() as conn:
        return conn.execute(
            "DELETE FROM ask_requests WHERE started_at <= %s",
            (cutoff,),
        ).rowcount

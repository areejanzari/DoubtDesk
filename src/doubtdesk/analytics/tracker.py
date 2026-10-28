import logging
import sqlite3
from pathlib import Path
from doubtdesk.analytics.store import DB_PATH

logger = logging.getLogger(__name__)


def record_gemini_usage(
    *,
    response,
    request_id: str,
    conversation_id: str | None,
    operation: str,
    model: str,
    latency_ms: int,
    db_path: Path = DB_PATH,
) -> None:
    usage = getattr(response, "usage_metadata", None)

    if usage is None:
        logger.warning(
            "Gemini usage metadata unavailable for %s",
            operation,
        )
        return

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO ai_usage (
                    request_id,
                    conversation_id,
                    operation,
                    model,
                    prompt_tokens,
                    output_tokens,
                    thinking_tokens,
                    cached_tokens,
                    total_tokens,
                    latency_ms,
                    status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    conversation_id,
                    operation,
                    model,
                    usage.prompt_token_count,
                    usage.candidates_token_count,
                    usage.thoughts_token_count,
                    usage.cached_content_token_count,
                    usage.total_token_count,
                    latency_ms,
                    "success",
                ),
            )
    except sqlite3.Error:
        logger.exception("Failed to save Gemini usage analytics")


def record_gemini_failure(
    *,
    request_id: str,
    conversation_id: str | None,
    operation: str,
    model: str,
    latency_ms: int,
    error_type: str,
    response=None,
    db_path: Path = DB_PATH,
) -> None:
    usage = getattr(response, "usage_metadata", None)

    def token_count(field: str):
        return getattr(usage, field, None)

    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO ai_usage (
                    request_id,
                    conversation_id,
                    operation,
                    model,
                    prompt_tokens,
                    output_tokens,
                    thinking_tokens,
                    cached_tokens,
                    total_tokens,
                    latency_ms,
                    status,
                    error_type
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request_id,
                    conversation_id,
                    operation,
                    model,
                    token_count("prompt_token_count"),
                    token_count("candidates_token_count"),
                    token_count("thoughts_token_count"),
                    token_count("cached_content_token_count"),
                    token_count("total_token_count"),
                    latency_ms,
                    "error",
                    error_type,
                ),
            )
    except sqlite3.Error:
        logger.exception("Failed to save Gemini error analytics.")
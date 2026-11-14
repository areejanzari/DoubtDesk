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
    pacer_wait_ms: int | None = None,
    db_path: Path = DB_PATH,
) -> None:
    usage = getattr(response, "usage_metadata", None)


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
                    status, provider, pacer_wait_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'gemini', ?)
                """,
                (
                    request_id,
                    conversation_id,
                    operation,
                    model,
                    getattr(usage, "prompt_token_count", None),
                    getattr(usage, "candidates_token_count", None),
                    getattr(usage, "thoughts_token_count", None),
                    getattr(usage, "cached_content_token_count", None),
                    getattr(usage, "total_token_count", None),
                    latency_ms,
                    "success",
                    pacer_wait_ms,
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
    pacer_wait_ms: int | None = None,
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
                    error_type, provider, pacer_wait_ms
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'gemini', ?)
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
                    pacer_wait_ms,
                ),
            )
    except sqlite3.Error:
        logger.exception("Failed to save Gemini error analytics.")

def record_openai_usage(*, response, request_id, conversation_id=None, operation, model,
                        latency_ms=None, db_path=DB_PATH):
    usage = getattr(response, "usage", None)
    if usage is None or not request_id:
        return
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """INSERT INTO ai_usage
                    (request_id, conversation_id, operation, model, prompt_tokens,
                     output_tokens, thinking_tokens, cached_tokens, total_tokens,
                     latency_ms, status, provider)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'success', 'openai')""",
                (
                    request_id, conversation_id, operation, model,
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                    getattr(output_details, "reasoning_tokens", None),
                    getattr(input_details, "cached_tokens", None),
                    getattr(usage, "total_tokens", None), latency_ms,
                ),
            )
    except sqlite3.Error:
        logger.exception("Failed to record OpenAI usage")

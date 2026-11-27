import logging
import psycopg
from doubtdesk.db import connection

logger = logging.getLogger(__name__)


def record_openai_usage(*, response, request_id, conversation_id=None, operation, model,
                        latency_ms=None):
    usage = getattr(response, "usage", None)
    if usage is None or not request_id:
        return
    input_details = getattr(usage, "input_tokens_details", None)
    output_details = getattr(usage, "output_tokens_details", None)
    try:
        with connection() as conn:
            conn.execute(
                """INSERT INTO ai_usage
                    (request_id, conversation_id, operation, model, prompt_tokens,
                     output_tokens, thinking_tokens, cached_tokens, total_tokens,
                     latency_ms, status, provider)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'success', 'openai')""",
                (
                    request_id, conversation_id, operation, model,
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                    getattr(output_details, "reasoning_tokens", None),
                    getattr(input_details, "cached_tokens", None),
                    getattr(usage, "total_tokens", None), latency_ms,
                ),
            )
    except psycopg.Error:
        logger.exception("Failed to record OpenAI usage")


def _anthropic_usage_row(response):
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None, None
    prompt = getattr(usage, "input_tokens", None)
    if prompt is None:
        prompt = getattr(usage, "prompt_tokens", None)
    output = getattr(usage, "output_tokens", None)
    if output is None:
        output = getattr(usage, "completion_tokens", None)
    cached = getattr(usage, "cache_read_input_tokens", None)
    total = (prompt or 0) + (output or 0) if prompt is not None or output is not None else None
    return prompt, output, cached, total


def record_anthropic_usage(*, response, request_id, conversation_id=None, operation, model,
                           latency_ms=None, pacer_wait_ms=None, provider="bedrock", status="success",
                           error_type=None):
    if not request_id:
        return
    prompt, output, cached, total = _anthropic_usage_row(response)
    try:
        with connection() as conn:
            conn.execute(
                """INSERT INTO ai_usage
                    (request_id, conversation_id, operation, model, prompt_tokens,
                     output_tokens, cached_tokens, total_tokens, latency_ms, status,
                     error_type, provider, pacer_wait_ms)
                     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (request_id, conversation_id, operation, model, prompt, output, cached, total,
                 latency_ms, status, error_type, provider, pacer_wait_ms),
            )
    except psycopg.Error:
        logger.exception("Failed to record Anthropic usage")


def record_anthropic_failure(*, request_id, conversation_id=None, operation, model, latency_ms,
                             error_type, response=None, pacer_wait_ms=None, provider="bedrock"):
    record_anthropic_usage(response=response, request_id=request_id, conversation_id=conversation_id,
                           operation=operation, model=model, latency_ms=latency_ms,
                           pacer_wait_ms=pacer_wait_ms, provider=provider, status="error",
                           error_type=error_type)

from types import SimpleNamespace
from doubtdesk.db import connection
from doubtdesk.analytics.queries import get_usage_overview, get_request_usage
from doubtdesk.analytics.tracker import record_anthropic_usage, record_anthropic_failure
from doubtdesk.analytics.queries import get_daily_usage


def test_empty_analytics_database():
    result = get_usage_overview()

    assert result["model_calls"] == 0
    assert result["total_requests"] == 0
    assert result["total_tokens"] == 0
    assert result["failure_rate"] == 0
    assert result["by_operation"] == []
    assert result["common_errors"] == []


def test_analytics_failure_tracking():
    record_anthropic_failure(
        request_id="request_001",
        conversation_id="chat_001",
        operation="vision",
        model="test-model",
        latency_ms=1500,
        error_type="TimeoutError",
    )

    result = get_usage_overview()

    assert result["model_calls"] == 1
    assert result["total_requests"] == 1
    assert result["successful_calls"] == 0
    assert result["failed_calls"] == 1
    assert result["failure_rate"] == 100.0
    assert result["total_tokens"] == 0

    assert result["common_errors"] == [
        {"error_type": "TimeoutError", "count": 1}
    ]

def test_mixed_model_usage():
    success_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=200,
            completion_tokens=100,
        )
    )

    failure_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=400,
            completion_tokens=200,
        )
    )

    record_anthropic_usage(
        response=success_response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="paraphrasing",
        model="test-model",
        latency_ms=1000,
    )

    record_anthropic_failure(
        response=failure_response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="vision",
        model="test-model",
        latency_ms=3000,
        error_type="ValidationError",
    )

    result = get_usage_overview()

    assert result["model_calls"] == 2
    assert result["total_requests"] == 1
    assert result["successful_calls"] == 1
    assert result["failed_calls"] == 1
    assert result["failure_rate"] == 50.0
    assert result["total_tokens"] == 900
    assert result["average_latency_ms"] == 2000.0

def test_analytics_date_filtering():
    for request_id in ["recent_request", "old_request"]:
        record_anthropic_failure(
            request_id=request_id,
            conversation_id=None,
            operation="vision",
            model="test-model",
            latency_ms=1000,
            error_type="TimeoutError",
        )

    with connection() as conn:
        conn.execute(
            """
            UPDATE ai_usage
            SET created_at = now() - interval '10 days'
            WHERE request_id = %s
            """,
            ("old_request",),
        )

    all_time = get_usage_overview()
    last_7_days = get_usage_overview(days=7)
    last_30_days = get_usage_overview(days=30)

    assert all_time["model_calls"] == 2
    assert last_7_days["model_calls"] == 1
    assert last_30_days["model_calls"] == 2


def test_request_usage_includes_failures():
    success_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
        )
    )

    record_anthropic_usage(
        response=success_response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="paraphrasing",
        model="test-model",
        latency_ms=1000,
    )

    record_anthropic_failure(
        request_id="request_001",
        conversation_id="chat_001",
        operation="vision",
        model="test-model",
        latency_ms=2000,
        error_type="TimeoutError",
    )

    record_anthropic_failure(
        request_id="request_002",
        conversation_id="chat_002",
        operation="paraphrasing",
        model="test-model",
        latency_ms=500,
        error_type="TimeoutError",
    )

    result = get_request_usage()

    assert result["total_requests"] == 2

    requests = {
        item["request_id"]: item
        for item in result["requests"]
    }

    partial = requests["request_001"]

    assert partial["model_calls"] == 2
    assert partial["successful_calls"] == 1
    assert partial["failed_calls"] == 1
    assert partial["total_tokens"] == 150
    assert partial["model_latency_ms"] == 3000
    assert partial["request_status"] == "partial_failure"

    failed = requests["request_002"]

    assert failed["model_calls"] == 1
    assert failed["successful_calls"] == 0
    assert failed["failed_calls"] == 1
    assert failed["request_status"] == "failed"


def test_daily_usage():
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=50,
        )
    )

    record_anthropic_usage(
        response=response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="paraphrasing",
        model="test-model",
        latency_ms=500,
    )

    record_anthropic_failure(
        request_id="request_002",
        conversation_id="chat_002",
        operation="vision",
        model="test-model",
        latency_ms=1000,
        error_type="TimeoutError",
    )

    result = get_daily_usage(days=7)

    assert len(result) == 2

    operations = {row["operation"]: row for row in result}

    assert operations["paraphrasing"]["total_tokens"] == 150
    assert operations["paraphrasing"]["successful_calls"] == 1
    assert operations["vision"]["failed_calls"] == 1
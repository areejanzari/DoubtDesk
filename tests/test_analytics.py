import sqlite3
from types import SimpleNamespace
from doubtdesk.analytics.store import initialize_analytics_db
from doubtdesk.analytics.queries import get_usage_overview, get_request_usage
from doubtdesk.analytics.tracker import record_gemini_usage, record_gemini_failure
from doubtdesk.analytics.queries import get_daily_usage


def test_empty_analytics_database(tmp_path):
    db_path = tmp_path / "analytics.db"
    initialize_analytics_db(db_path)

    result = get_usage_overview(db_path=db_path)

    assert result["gemini_calls"] == 0
    assert result["total_requests"] == 0
    assert result["total_tokens"] == 0
    assert result["failure_rate"] == 0
    assert result["by_operation"] == []
    assert result["common_errors"] == []


def test_analytics_failure_tracking(tmp_path):
    db_path = tmp_path / "analytics.db"
    initialize_analytics_db(db_path)

    record_gemini_failure(
        request_id="request_001",
        conversation_id="chat_001",
        operation="vision",
        model="test-model",
        latency_ms=1500,
        error_type="TimeoutError",
        db_path=db_path,
    )

    result = get_usage_overview(db_path=db_path)

    assert result["gemini_calls"] == 1
    assert result["total_requests"] == 1
    assert result["successful_calls"] == 0
    assert result["failed_calls"] == 1
    assert result["failure_rate"] == 100.0
    assert result["total_tokens"] == 0

    assert result["common_errors"] == [
        {"error_type": "TimeoutError", "count": 1}
    ]

def test_mixed_gemini_usage(tmp_path):
    db_path = tmp_path / "analytics.db"
    initialize_analytics_db(db_path)

    success_response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=200,
            candidates_token_count=100,
            thoughts_token_count=None,
            cached_content_token_count=0,
            total_token_count=300,
        )
    )

    failure_response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=400,
            candidates_token_count=150,
            thoughts_token_count=50,
            cached_content_token_count=0,
            total_token_count=600,
        )
    )

    record_gemini_usage(
        response=success_response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="paraphrasing",
        model="test-model",
        latency_ms=1000,
        db_path=db_path,
    )

    record_gemini_failure(
        response=failure_response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="vision",
        model="test-model",
        latency_ms=3000,
        error_type="ValidationError",
        db_path=db_path,
    )

    result = get_usage_overview(db_path=db_path)

    assert result["gemini_calls"] == 2
    assert result["total_requests"] == 1
    assert result["successful_calls"] == 1
    assert result["failed_calls"] == 1
    assert result["failure_rate"] == 50.0
    assert result["total_tokens"] == 900
    assert result["average_latency_ms"] == 2000.0

def test_analytics_date_filtering(tmp_path):
    db_path = tmp_path / "analytics.db"
    initialize_analytics_db(db_path)

    for request_id in ["recent_request", "old_request"]:
        record_gemini_failure(
            request_id=request_id,
            conversation_id=None,
            operation="vision",
            model="test-model",
            latency_ms=1000,
            error_type="TimeoutError",
            db_path=db_path,
        )

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE ai_usage
            SET created_at = datetime('now', '-10 days')
            WHERE request_id = ?
            """,
            ("old_request",),
        )

    all_time = get_usage_overview(db_path=db_path)
    last_7_days = get_usage_overview(db_path=db_path, days=7)
    last_30_days = get_usage_overview(db_path=db_path, days=30)

    assert all_time["gemini_calls"] == 2
    assert last_7_days["gemini_calls"] == 1
    assert last_30_days["gemini_calls"] == 2


def test_request_usage_includes_failures(tmp_path):
    db_path = tmp_path / "analytics.db"
    initialize_analytics_db(db_path)

    success_response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            thoughts_token_count=None,
            cached_content_token_count=0,
            total_token_count=150,
        )
    )

    record_gemini_usage(
        response=success_response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="paraphrasing",
        model="test-model",
        latency_ms=1000,
        db_path=db_path,
    )

    record_gemini_failure(
        request_id="request_001",
        conversation_id="chat_001",
        operation="vision",
        model="test-model",
        latency_ms=2000,
        error_type="TimeoutError",
        db_path=db_path,
    )

    record_gemini_failure(
        request_id="request_002",
        conversation_id="chat_002",
        operation="paraphrasing",
        model="test-model",
        latency_ms=500,
        error_type="TimeoutError",
        db_path=db_path,
    )

    result = get_request_usage(db_path=db_path)

    assert result["total_requests"] == 2

    requests = {
        item["request_id"]: item
        for item in result["requests"]
    }

    partial = requests["request_001"]

    assert partial["gemini_calls"] == 2
    assert partial["successful_calls"] == 1
    assert partial["failed_calls"] == 1
    assert partial["total_tokens"] == 150
    assert partial["gemini_latency_ms"] == 3000
    assert partial["request_status"] == "partial_failure"

    failed = requests["request_002"]

    assert failed["gemini_calls"] == 1
    assert failed["successful_calls"] == 0
    assert failed["failed_calls"] == 1
    assert failed["request_status"] == "failed"


def test_daily_usage(tmp_path):
    db_path = tmp_path / "analytics.db"
    initialize_analytics_db(db_path)

    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=50,
            thoughts_token_count=None,
            cached_content_token_count=0,
            total_token_count=150,
        )
    )

    record_gemini_usage(
        response=response,
        request_id="request_001",
        conversation_id="chat_001",
        operation="paraphrasing",
        model="test-model",
        latency_ms=500,
        db_path=db_path,
    )

    record_gemini_failure(
        request_id="request_002",
        conversation_id="chat_002",
        operation="vision",
        model="test-model",
        latency_ms=1000,
        error_type="TimeoutError",
        db_path=db_path,
    )

    result = get_daily_usage(db_path=db_path, days=7)

    assert len(result) == 2

    operations = {row["operation"]: row for row in result}

    assert operations["paraphrasing"]["total_tokens"] == 150
    assert operations["paraphrasing"]["successful_calls"] == 1
    assert operations["vision"]["failed_calls"] == 1
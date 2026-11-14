from types import SimpleNamespace
import sqlite3
from doubtdesk.analytics.store import initialize_analytics_db
from doubtdesk.analytics.tracker import record_gemini_usage, record_gemini_failure, record_openai_usage
from doubtdesk.analytics.queries import get_usage_overview, get_request_usage


def test_provider_metrics_and_missing_usage_are_recorded(tmp_path):
    db = tmp_path / 'analytics.db'
    initialize_analytics_db(db)
    record_gemini_usage(response=SimpleNamespace(), request_id='r', conversation_id='c', operation='output_judge', model='gemini-test', latency_ms=10, pacer_wait_ms=1500, db_path=db)
    record_gemini_failure(request_id='r', conversation_id='c', operation='output_judge', model='gemini-test', latency_ms=20, error_type='TimeoutError', pacer_wait_ms=5, db_path=db)
    record_openai_usage(response=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=3, total_tokens=13)), request_id='r', operation='general_answer', model='openai-test', db_path=db)
    with sqlite3.connect(db) as conn:
        assert conn.execute('SELECT provider,status,pacer_wait_ms FROM ai_usage ORDER BY id').fetchall() == [('gemini','success',1500),('gemini','error',5),('openai','success',None)]
    overview = get_usage_overview(db)
    assert overview['model_calls'] == overview['gemini_calls'] == 3
    assert overview['gemini_judge_calls'] == 2
    assert overview['judge_wait_ms'] == 1505
    row = get_request_usage(db)['requests'][0]
    assert row['model_calls'] == 3 and row['judge_wait_ms'] == 1505

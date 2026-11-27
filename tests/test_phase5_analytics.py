from types import SimpleNamespace
from doubtdesk.db import connection
from doubtdesk.analytics.tracker import record_anthropic_usage, record_anthropic_failure, record_openai_usage
from doubtdesk.analytics.queries import get_usage_overview, get_request_usage


def test_provider_metrics_and_missing_usage_are_recorded():
    record_anthropic_usage(response=SimpleNamespace(), request_id='r', conversation_id='c', operation='output_judge', model='judge-test', latency_ms=10, pacer_wait_ms=1500, provider='bedrock_openai')
    record_anthropic_failure(request_id='r', conversation_id='c', operation='output_judge', model='judge-test', latency_ms=20, error_type='TimeoutError', pacer_wait_ms=5, provider='bedrock_openai')
    record_openai_usage(response=SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=3, total_tokens=13)), request_id='r', operation='general_answer', model='openai-test')
    with connection() as conn:
        assert conn.execute('SELECT provider,status,pacer_wait_ms FROM ai_usage ORDER BY id').fetchall() == [('bedrock_openai','success',1500),('bedrock_openai','error',5),('openai','success',None)]
    overview = get_usage_overview()
    assert overview['model_calls'] == 3
    assert overview['judge_calls'] == 2
    assert overview['judge_wait_ms'] == 1505
    row = get_request_usage()['requests'][0]
    assert row['model_calls'] == 3 and row['judge_wait_ms'] == 1505

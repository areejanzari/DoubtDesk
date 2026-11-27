from psycopg.rows import dict_row
from doubtdesk.db import TIMESTAMP_FORMAT, connection


def get_usage_overview(
    days: int | None = None,
) -> dict:
    where_clause = "WHERE TRUE"
    parameters = []

    if days is not None:
        where_clause += " AND created_at >= now() - make_interval(days => %s::int)"
        parameters.append(days)

    with connection() as conn:
        cursor = conn.cursor(row_factory=dict_row)

        summary = cursor.execute(
            f"""
            SELECT
                COUNT(*) AS model_calls,
                COALESCE(SUM(CASE WHEN operation = 'output_judge' THEN 1 ELSE 0 END), 0) AS judge_calls,
                COALESCE(SUM(pacer_wait_ms), 0) AS judge_wait_ms,
                COUNT(DISTINCT request_id) AS total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_calls,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)
                    AS failed_calls,
                COALESCE(
                    ROUND(
                        100.0 * SUM(
                            CASE WHEN status = 'error' THEN 1 ELSE 0 END
                        ) / NULLIF(COUNT(*), 0),
                        2
                    ),
                    0
                )::float8 AS failure_rate,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(ROUND(AVG(latency_ms), 1), 0)::float8
                    AS average_latency_ms
            FROM ai_usage
            {where_clause}
            """,
            parameters,
        ).fetchone()

        operations = cursor.execute(
            f"""
            SELECT
                operation,
                COUNT(*) AS calls,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_calls,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)
                    AS failed_calls,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(ROUND(AVG(latency_ms), 1), 0)::float8
                    AS average_latency_ms
            FROM ai_usage
            {where_clause}
            GROUP BY operation
            ORDER BY total_tokens DESC
            """,
            parameters,
        ).fetchall()

        errors = cursor.execute(
            f"""
            SELECT
                error_type,
                COUNT(*) AS count
            FROM ai_usage
            {where_clause}
                AND status = 'error'
            GROUP BY error_type
            ORDER BY count DESC
            LIMIT 10
            """,
            parameters,
        ).fetchall()

    return {
        **summary,
        "by_operation": operations,
        "common_errors": errors,
    }



def get_request_usage(
    days: int | None = None,
    limit: int = 20,
    page: int = 1,
) -> dict:
    where_clause = "WHERE TRUE"
    parameters = []

    if days is not None:
        where_clause += " AND created_at >= now() - make_interval(days => %s::int)"
        parameters.append(days)

    offset = (page - 1) * limit

    with connection() as conn:
        cursor = conn.cursor(row_factory=dict_row)

        total_requests = cursor.execute(
            f"""
            SELECT COUNT(DISTINCT request_id) AS total
            FROM ai_usage
            {where_clause}
            """,
            parameters,
        ).fetchone()["total"]

        rows = cursor.execute(
            f"""
            SELECT
                request_id,
                MAX(conversation_id) AS conversation_id,
                COUNT(*) AS model_calls,
                COALESCE(SUM(CASE WHEN operation = 'output_judge' THEN 1 ELSE 0 END), 0) AS judge_calls,
                COALESCE(SUM(pacer_wait_ms), 0) AS judge_wait_ms,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_calls,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)
                    AS failed_calls,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(SUM(latency_ms), 0) AS model_latency_ms,
                string_agg(DISTINCT operation, ',') AS operations,
                to_char(MAX(created_at) AT TIME ZONE 'UTC', '{TIMESTAMP_FORMAT}') AS created_at
            FROM ai_usage
            {where_clause}
            GROUP BY request_id
            ORDER BY MAX(created_at) DESC, MAX(id) DESC
            LIMIT %s OFFSET %s
            """,
            [*parameters, limit, offset],
        ).fetchall()

    requests = []

    for item in rows:
        item["operations"] = item["operations"].split(",")

        if item["failed_calls"] == 0:
            item["request_status"] = "success"
        elif item["successful_calls"] == 0:
            item["request_status"] = "failed"
        else:
            item["request_status"] = "partial_failure"

        requests.append(item)

    return {
        "total_requests": total_requests,
        "page": page,
        "limit": limit,
        "requests": requests,
    }



def get_daily_usage(
    days: int = 7,
) -> list[dict]:
    with connection() as conn:
        return conn.cursor(row_factory=dict_row).execute(
            """
            SELECT
                to_char((created_at AT TIME ZONE 'UTC')::date, 'YYYY-MM-DD') AS date,
                operation,
                COUNT(*) AS calls,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_calls,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)
                    AS failed_calls,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens
            FROM ai_usage
            WHERE created_at >= now() - make_interval(days => %s::int)
            GROUP BY 1, operation
            ORDER BY date ASC, operation ASC
            """,
            (days,),
        ).fetchall()

import sqlite3
from pathlib import Path
from doubtdesk.analytics.store import DB_PATH


def get_usage_overview(
    db_path: Path = DB_PATH,
    days: int | None = None,
) -> dict:
    where_clause = "WHERE 1=1"
    parameters = []

    if days is not None:
        where_clause += " AND created_at >= datetime('now', ?)"
        parameters.append(f"-{days} days")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        summary = conn.execute(
            f"""
            SELECT
                COUNT(*) AS gemini_calls,
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
                ) AS failure_rate,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(thinking_tokens), 0) AS thinking_tokens,
                COALESCE(SUM(cached_tokens), 0) AS cached_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(ROUND(AVG(latency_ms), 1), 0)
                    AS average_latency_ms
            FROM ai_usage
            {where_clause}
            """,
            parameters,
        ).fetchone()

        operations = conn.execute(
            f"""
            SELECT
                operation,
                COUNT(*) AS calls,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_calls,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)
                    AS failed_calls,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(ROUND(AVG(latency_ms), 1), 0)
                    AS average_latency_ms
            FROM ai_usage
            {where_clause}
            GROUP BY operation
            ORDER BY total_tokens DESC
            """,
            parameters,
        ).fetchall()

        errors = conn.execute(
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
        **dict(summary),
        "by_operation": [dict(row) for row in operations],
        "common_errors": [dict(row) for row in errors],
    }



def get_request_usage(
    db_path: Path = DB_PATH,
    days: int | None = None,
    limit: int = 20,
    page: int = 1,
) -> dict:
    where_clause = "WHERE 1=1"
    parameters = []

    if days is not None:
        where_clause += " AND created_at >= datetime('now', ?)"
        parameters.append(f"-{days} days")

    offset = (page - 1) * limit

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        total_requests = conn.execute(
            f"""
            SELECT COUNT(DISTINCT request_id)
            FROM ai_usage
            {where_clause}
            """,
            parameters,
        ).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT
                request_id,
                MAX(conversation_id) AS conversation_id,
                COUNT(*) AS gemini_calls,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END)
                    AS successful_calls,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END)
                    AS failed_calls,
                COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(total_tokens), 0) AS total_tokens,
                COALESCE(SUM(latency_ms), 0) AS gemini_latency_ms,
                GROUP_CONCAT(DISTINCT operation) AS operations,
                MAX(created_at) AS created_at,
                MAX(id) AS last_record_id
            FROM ai_usage
            {where_clause}
            GROUP BY request_id
            ORDER BY created_at DESC, last_record_id DESC
            LIMIT ? OFFSET ?
            """,
            [*parameters, limit, offset],
        ).fetchall()

    requests = []

    for row in rows:
        item = dict(row)
        item.pop("last_record_id")

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
    db_path: Path = DB_PATH,
    days: int = 7,
) -> list[dict]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT
                DATE(created_at) AS date,
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
            WHERE created_at >= datetime('now', ?)
            GROUP BY DATE(created_at), operation
            ORDER BY date ASC, operation ASC
            """,
            (f"-{days} days",),
        ).fetchall()

    return [dict(row) for row in rows]

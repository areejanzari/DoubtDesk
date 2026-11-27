import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from doubtdesk.db import connection


def _database_name(url: str) -> str:
    name = conninfo_to_dict(url).get("dbname", "")
    if not name.endswith(("_test", "_e2e")):
        raise RuntimeError(f"Refusing to use database {name!r} for tests; its name must end in _test or _e2e")
    return name


def ensure_database(url: str) -> None:
    name = _database_name(url)
    with psycopg.connect(make_conninfo(url, dbname="postgres"), autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone() is None:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))


def reset_database(url: str) -> None:
    _database_name(url)
    with connection() as conn:
        tables = [row[0] for row in conn.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")]
        if tables:
            conn.execute(
                sql.SQL("TRUNCATE {} RESTART IDENTITY CASCADE").format(
                    sql.SQL(", ").join(sql.Identifier(table) for table in tables)
                )
            )

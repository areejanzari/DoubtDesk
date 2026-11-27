import logging
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from threading import Lock

import psycopg
from psycopg_pool import ConnectionPool

from doubtdesk.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 384
TIMESTAMP_FORMAT = "YYYY-MM-DD HH24:MI:SS"

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS citext;

CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    page_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS processing_started_at BIGINT;

CREATE TABLE IF NOT EXISTS user_documents (
    user_id TEXT NOT NULL,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id),
    filename TEXT NOT NULL,
    attached_at BIGINT NOT NULL,
    PRIMARY KEY (user_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_user_documents_doc ON user_documents(doc_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', text)) STORED
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING GIN (tsv);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv_en tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;
CREATE INDEX IF NOT EXISTS idx_chunks_tsv_en ON chunks USING GIN (tsv_en);
CREATE INDEX IF NOT EXISTS idx_chunks_position ON chunks(doc_id, page_number, chunk_index);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw ON chunks USING hnsw (embedding vector_cosine_ops);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS chunker_version INTEGER NOT NULL DEFAULT 1;

CREATE TABLE IF NOT EXISTS pages (
    doc_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT NOT NULL,
    classification TEXT,
    ocr_status TEXT NOT NULL DEFAULT 'not_needed',
    figure_area REAL,
    figure_refs JSONB NOT NULL DEFAULT '[]',
    inline_image_count INTEGER,
    has_table BOOLEAN,
    watermark_detected BOOLEAN,
    visual BOOLEAN,
    PRIMARY KEY (doc_id, page_number)
);

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_accessed_ns BIGINT NOT NULL DEFAULT 0,
    title TEXT NOT NULL DEFAULT 'New chat',
    custom_title BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_conversations_activity
    ON conversations(user_id, last_accessed_ns DESC);

CREATE TABLE IF NOT EXISTS evicted_conversations (
    conversation_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    evicted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS exchanges (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    doc_ids JSONB NOT NULL,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    formatted_answer TEXT,
    sources JSONB NOT NULL DEFAULT '[]',
    memory_eligible BOOLEAN NOT NULL DEFAULT TRUE,
    memory_embedding vector({EMBEDDING_DIMENSIONS})
);
CREATE INDEX IF NOT EXISTS idx_exchanges_conversation ON exchanges(conversation_id, id);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username CITEXT NOT NULL UNIQUE,
    email CITEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    verified_at BIGINT,
    created_at BIGINT NOT NULL
);
ALTER TABLE users ADD COLUMN IF NOT EXISTS plan TEXT NOT NULL DEFAULT 'free'
    CHECK (plan IN ('free', 'pro', 'unlimited'));

CREATE TABLE IF NOT EXISTS ask_requests (
    request_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('in_flight', 'completed', 'failed')),
    started_at BIGINT NOT NULL,
    finished_at BIGINT
);
CREATE INDEX IF NOT EXISTS idx_ask_requests_user_time ON ask_requests(user_id, started_at);
CREATE INDEX IF NOT EXISTS idx_ask_requests_status ON ask_requests(user_id, status, started_at);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    csrf_token TEXT NOT NULL,
    expires_at BIGINT NOT NULL,
    created_at BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

CREATE TABLE IF NOT EXISTS action_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    purpose TEXT NOT NULL CHECK (purpose IN ('verify', 'reset')),
    expires_at BIGINT NOT NULL,
    created_at BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_action_tokens_user ON action_tokens(user_id, purpose);

CREATE TABLE IF NOT EXISTS login_failures (
    identifier_hash TEXT NOT NULL,
    ip_address TEXT NOT NULL,
    attempted_at BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_failures_time ON login_failures(attempted_at);
CREATE INDEX IF NOT EXISTS idx_login_failures_identifier ON login_failures(identifier_hash, attempted_at);
CREATE INDEX IF NOT EXISTS idx_login_failures_ip ON login_failures(ip_address, attempted_at);

CREATE TABLE IF NOT EXISTS auth_attempts (
    action TEXT NOT NULL,
    ip_address TEXT NOT NULL,
    email_hash TEXT NOT NULL,
    attempted_at BIGINT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_ip ON auth_attempts(action, ip_address, attempted_at);
CREATE INDEX IF NOT EXISTS idx_auth_attempts_email ON auth_attempts(action, email_hash, attempted_at);

CREATE TABLE IF NOT EXISTS ai_usage (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    request_id TEXT NOT NULL,
    conversation_id TEXT,
    operation TEXT NOT NULL,
    model TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'unknown',
    prompt_tokens INTEGER,
    output_tokens INTEGER,
    thinking_tokens INTEGER,
    cached_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER,
    pacer_wait_ms INTEGER,
    status TEXT NOT NULL,
    error_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_ai_usage_request ON ai_usage(request_id);
CREATE INDEX IF NOT EXISTS idx_ai_usage_created ON ai_usage(created_at);

CREATE TABLE IF NOT EXISTS eval_runs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0,
    total INTEGER NOT NULL,
    results JSONB NOT NULL DEFAULT '[]',
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

_SCHEMA_LOCK_KEY = 7_204_311_955

_pool: ConnectionPool | None = None
_pool_lock = Lock()


def _get_pool() -> ConnectionPool:
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = ConnectionPool(
                settings.database_url,
                min_size=settings.db_pool_min_size,
                max_size=settings.db_pool_max_size,
                max_idle=300,
                check=ConnectionPool.check_connection,
                open=True,
            )
        return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    with _get_pool().connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


def initialize_database(timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with psycopg.connect(settings.database_url, autocommit=True) as conn:
                conn.execute("SELECT pg_advisory_lock(%s)", (_SCHEMA_LOCK_KEY,))
                try:
                    conn.execute(SCHEMA)
                finally:
                    conn.execute("SELECT pg_advisory_unlock(%s)", (_SCHEMA_LOCK_KEY,))
            return
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                raise
            logger.warning("Database not reachable yet; retrying")
            time.sleep(2)


def to_vector(values: Sequence[float]) -> str:
    return "[" + ",".join(repr(float(value)) for value in values) + "]"

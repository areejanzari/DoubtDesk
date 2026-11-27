import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    openai_api_key: str = ""
    openai_model: str = "gpt-6-luna"
    ragas_judge_model: str = "qwen.qwen3-235b-a22b-2507"
    database_url: str = "postgresql://doubtdesk:doubtdesk@127.0.0.1:5432/doubtdesk"
    db_pool_min_size: int = Field(default=1, ge=1)
    db_pool_max_size: int = Field(default=20, ge=1)
    model_threads: int = Field(default_factory=lambda: max(1, min(4, (os.cpu_count() or 2) // 2)), ge=1)
    model_concurrency: int = Field(default=2, ge=1)
    model_cache_dir: Path = Path("data/models")
    ingestion_workers: int = Field(default=1, ge=1)
    upload_dir: Path = Path("data/uploads")
    page_image_dir: Path = Path("data/page_images")
    storage_backend: Literal["local", "s3"] = "local"
    s3_bucket_name: str = ""
    paraphrase_provider: Literal["openai", "bedrock_openai"] = "openai"
    paraphrase_model: str = ""
    evidence_assessment_model: str = ""
    output_judge_enabled: bool = True
    output_judge_provider: Literal["bedrock_openai", "bedrock"] = "bedrock_openai"
    output_judge_model: str = "qwen.qwen3-vl-235b-a22b-instruct"
    output_judge_fallback_model: str = ""
    output_judge_rpm: int = Field(default=120, ge=1)
    aws_region: str = ""
    output_judge_wait_log_ms: int = Field(default=1000, ge=0)
    judge_rpm: int = Field(default=5, ge=1)
    judge_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    evidence_text_threshold: float = Field(default=.8, ge=0, le=1)
    evidence_moderate_threshold: float = Field(default=.5, ge=0, le=1)
    evidence_assessment_enabled: bool = False
    rerank_low_threshold: float = -5.0
    ocr_language: str = "eng"
    ocr_dpi: int = Field(default=300, ge=72, le=600)
    retrieval_top_k: int = Field(default=6, ge=1)
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    evidence_forward_top_k: int = Field(default=6, ge=1)
    memory_max_distance: float = Field(default=.6, ge=0, le=2)
    max_pdf_pages: int = Field(default=300, ge=1)
    trusted_proxy_count: int = Field(default=0, ge=0)
    free_daily_questions: int = Field(default=30, ge=0)
    free_daily_tokens: int = Field(default=200_000, ge=0)
    free_max_concurrent: int = Field(default=1, ge=1)
    pro_daily_questions: int = Field(default=300, ge=0)
    pro_daily_tokens: int = Field(default=3_000_000, ge=0)
    pro_max_concurrent: int = Field(default=3, ge=1)
    inflight_stale_seconds: int = Field(default=600, ge=1)

settings = Settings()


def runtime_settings() -> Settings:
    return Settings(_env_file=None)

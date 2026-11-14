from pathlib import Path
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
    gemini_api_key: str = ""
    ragas_judge_model: str = "gemini-3.1-flash-lite"
    chroma_path: Path = Path("data/chroma")
    documents_db: Path = Path("data/documents.db")
    conversations_db: Path = Path("data/conversations.db")
    analytics_db: Path = Path("data/analytics.db")
    auth_db: Path = Path("data/auth.db")
    evals_db: Path = Path("data/evals.db")
    lexical_db: Path = Path("data/lexical.db")
    page_image_dir: Path = Path("data/page_images")
    paraphrase_model: str = ""
    evidence_assessment_model: str = ""
    output_judge_enabled: bool = True
    output_judge_model: str = "gemini-3.1-flash-lite"
    output_judge_fallback_model: str = ""
    output_judge_rpm: int = Field(default=5, ge=1)
    output_judge_wait_log_ms: int = Field(default=1000, ge=0)
    judge_rpm: int = Field(default=5, ge=1)
    judge_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    evidence_text_threshold: float = Field(default=.8, ge=0, le=1)
    evidence_moderate_threshold: float = Field(default=.5, ge=0, le=1)
    retrieval_top_k: int = Field(default=6, ge=1)
    evidence_forward_top_k: int = Field(default=4, ge=1)
    memory_max_distance: float = Field(default=.6, ge=0, le=2)
    max_pdf_pages: int = Field(default=300, ge=1)
    trusted_proxy_count: int = Field(default=0, ge=0)


# Database paths are fixed at process startup, so stores and routes agree.
settings = Settings()


def runtime_settings() -> Settings:
    return Settings(_env_file=None)

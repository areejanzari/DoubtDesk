from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict

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

settings = Settings()
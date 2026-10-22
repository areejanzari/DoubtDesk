from pathlib import Path
from pydantic_settings import BaseSettings,SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite-preview"
    chroma_path: Path = Path("data/chroma")

settings = Settings()
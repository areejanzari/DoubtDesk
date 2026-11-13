from pathlib import Path
import pytest
from doubtdesk.config import Settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for key in [
        "APP_ENV",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "GEMINI_API_KEY",
        "RAGAS_JUDGE_MODEL",
        "CHROMA_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_default_settings(clean_env):
    config = Settings()
    assert config.app_env == "development"
    assert config.openai_api_key == ""
    assert config.openai_model == "gpt-6-luna"
    assert config.gemini_api_key == ""
    assert config.ragas_judge_model == "gemini-3.1-flash-lite"
    assert config.chroma_path == Path("data/chroma")


def test_environment_variables(clean_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("OPENAI_MODEL", "test-openai-model")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "test-eval-model")
    monkeypatch.setenv("CHROMA_PATH", "custom/chroma")

    config = Settings()
    assert config.app_env == "production"
    assert config.openai_api_key == "test_openai_key"
    assert config.openai_model == "test-openai-model"
    assert config.gemini_api_key == "test_gemini_key"
    assert config.ragas_judge_model == "test-eval-model"
    assert config.chroma_path == Path("custom/chroma")


def test_dotenv_file(clean_env, tmp_path):
    (tmp_path / ".env").write_text(
        "APP_ENV=testing\nOPENAI_MODEL=dotenv-model\nRAGAS_JUDGE_MODEL=dotenv-eval\n",
        encoding="utf-8",
    )
    config = Settings()
    assert config.app_env == "testing"
    assert config.openai_model == "dotenv-model"
    assert config.ragas_judge_model == "dotenv-eval"


def test_environment_overrides_dotenv(clean_env, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text("OPENAI_MODEL=dotenv-model\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    assert Settings().openai_model == "environment-model"


def test_extra_environment_values_ignored(clean_env, tmp_path):
    (tmp_path / ".env").write_text(
        "APP_ENV=testing\nUNUSED_SETTING=hello\n", encoding="utf-8"
    )
    config = Settings()
    assert config.app_env == "testing"
    assert not hasattr(config, "unused_setting")


def test_custom_chroma_path(clean_env, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", "storage/vectors")
    config = Settings()
    assert isinstance(config.chroma_path, Path)
    assert config.chroma_path == Path("storage/vectors")

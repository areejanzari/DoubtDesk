from pathlib import Path
import pytest
from doubtdesk.config import Settings


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    for key in [
        "APP_ENV",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "CHROMA_PATH",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_default_settings(clean_env):
    config = Settings()

    assert config.app_env == "development"
    assert config.gemini_api_key == ""
    assert config.gemini_model == "gemini-3.1-flash-lite-preview"
    assert config.chroma_path == Path("data/chroma")


def test_environment_variables(clean_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("GEMINI_API_KEY", "test_api_key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("CHROMA_PATH", "custom/chroma")

    config = Settings()

    assert config.app_env == "production"
    assert config.gemini_api_key == "test_api_key"
    assert config.gemini_model == "test-model"
    assert config.chroma_path == Path("custom/chroma")


def test_dotenv_file(clean_env, tmp_path):
    env_file = tmp_path / ".env"

    env_file.write_text(
        "APP_ENV=testing\n"
        "GEMINI_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    config = Settings()

    assert config.app_env == "testing"
    assert config.gemini_model == "dotenv-model"


def test_environment_overrides_dotenv(clean_env, monkeypatch, tmp_path):
    (tmp_path / ".env").write_text(
        "GEMINI_MODEL=dotenv-model\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("GEMINI_MODEL", "environment-model")

    config = Settings()

    assert config.gemini_model == "environment-model"


def test_extra_environment_values_ignored(clean_env, tmp_path):
    (tmp_path / ".env").write_text(
        "APP_ENV=testing\n"
        "UNUSED_SETTING=hello\n",
        encoding="utf-8",
    )

    config = Settings()

    assert config.app_env == "testing"
    assert not hasattr(config, "unused_setting")


def test_custom_chroma_path(clean_env, monkeypatch):
    monkeypatch.setenv("CHROMA_PATH", "storage/vectors")

    config = Settings()

    assert isinstance(config.chroma_path, Path)
    assert config.chroma_path == Path("storage/vectors")
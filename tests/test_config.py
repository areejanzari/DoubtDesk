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
        "RAGAS_JUDGE_MODEL",
        "DATABASE_URL",
    ]:
        monkeypatch.delenv(key, raising=False)


def test_default_settings(clean_env):
    config = Settings()
    assert config.app_env == "development"
    assert config.openai_api_key == ""
    assert config.openai_model == "gpt-6-luna"
    assert config.ragas_judge_model == "qwen.qwen3-235b-a22b-2507"
    assert config.database_url == "postgresql://doubtdesk:doubtdesk@127.0.0.1:5432/doubtdesk"


def test_environment_variables(clean_env, monkeypatch):
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    monkeypatch.setenv("OPENAI_MODEL", "test-openai-model")
    monkeypatch.setenv("RAGAS_JUDGE_MODEL", "test-eval-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://app:secret@10.0.0.5:5432/doubtdesk")

    config = Settings()
    assert config.app_env == "production"
    assert config.openai_api_key == "test_openai_key"
    assert config.openai_model == "test-openai-model"
    assert config.ragas_judge_model == "test-eval-model"
    assert config.database_url == "postgresql://app:secret@10.0.0.5:5432/doubtdesk"


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


def test_custom_page_image_dir(clean_env, monkeypatch):
    monkeypatch.setenv("PAGE_IMAGE_DIR", "storage/pages")
    config = Settings()
    assert isinstance(config.page_image_dir, Path)
    assert config.page_image_dir == Path("storage/pages")

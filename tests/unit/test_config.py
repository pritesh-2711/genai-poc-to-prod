"""Unit tests for ConfigManager."""

import os
import pytest
from pathlib import Path

from src.core.config import ConfigManager
from src.core.exceptions import ConfigurationError

# ---------------------------------------------------------------------------
# Minimal valid YAML that satisfies all required fields
# ---------------------------------------------------------------------------

_BASE_YAML = """\
embeddings:
  provider: local
  local:
    model: all-MiniLM-L6-v2
    dimension: 384
llm:
  provider: openai
  openai:
    model: gpt-4.1-mini
    api_key: sk-test
database:
  host: localhost
  port: 5432
  database: testdb
  user: testuser
  password: testpass
storage:
  deployment: local
guardrails:
  enabled: false
reranker:
  enabled: false
  model: BAAI/bge-reranker-base
  top_k: 5
  device: cpu
mcp:
  enabled: false
  transport: stdio
  stdio:
    command: python
    args: []
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(_BASE_YAML)
    return p


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_loads_yaml_file(config_path: Path) -> None:
    cm = ConfigManager(str(config_path))
    assert cm.config is not None


def test_missing_config_file_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        ConfigManager("/nonexistent/path/config.yaml")


# ---------------------------------------------------------------------------
# Env-var resolution
# ---------------------------------------------------------------------------

def test_env_var_placeholder_is_resolved(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_SECRET_VALUE", "super-secret")
    cm = ConfigManager(str(config_path))
    result = cm._resolve_env_vars("${TEST_SECRET_VALUE}")
    assert result == "super-secret"


def test_unset_env_var_returns_placeholder_string(config_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEFINITELY_NOT_SET_VAR", raising=False)
    cm = ConfigManager(str(config_path))
    result = cm._resolve_env_vars("${DEFINITELY_NOT_SET_VAR}")
    assert result == "${DEFINITELY_NOT_SET_VAR}"


def test_non_placeholder_string_is_returned_unchanged(config_path: Path) -> None:
    cm = ConfigManager(str(config_path))
    assert cm._resolve_env_vars("plain-value") == "plain-value"


# ---------------------------------------------------------------------------
# Nested key access via .get()
# ---------------------------------------------------------------------------

def test_nested_key_access(config_path: Path) -> None:
    cm = ConfigManager(str(config_path))
    assert cm.get("embeddings.provider") == "local"


def test_missing_optional_key_returns_default(config_path: Path) -> None:
    cm = ConfigManager(str(config_path))
    assert cm.get("nonexistent.key", "default_value") == "default_value"


def test_deeply_nested_key_access(config_path: Path) -> None:
    cm = ConfigManager(str(config_path))
    assert cm.get("embeddings.local.dimension") == 384


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_unsupported_llm_provider_raises(tmp_path: Path) -> None:
    yaml = _BASE_YAML.replace("provider: openai\n  openai:", "provider: unsupported\n  unsupported:")
    p = tmp_path / "config.yaml"
    p.write_text(yaml)
    with pytest.raises(ConfigurationError, match="Unsupported LLM provider"):
        ConfigManager(str(p))


def test_missing_embedding_model_raises(tmp_path: Path) -> None:
    yaml = _BASE_YAML.replace("model: all-MiniLM-L6-v2", "")
    p = tmp_path / "config.yaml"
    p.write_text(yaml)
    with pytest.raises(ConfigurationError, match="model is required"):
        ConfigManager(str(p))


def test_missing_embedding_dimension_raises(tmp_path: Path) -> None:
    yaml = _BASE_YAML.replace("dimension: 384", "")
    p = tmp_path / "config.yaml"
    p.write_text(yaml)
    with pytest.raises(ConfigurationError, match="dimension is required"):
        ConfigManager(str(p))


def test_unsupported_embedding_provider_raises(tmp_path: Path) -> None:
    yaml = _BASE_YAML.replace("provider: local", "provider: unknown_provider", 1)
    p = tmp_path / "config.yaml"
    p.write_text(yaml)
    with pytest.raises(ConfigurationError, match="Unsupported embedding provider"):
        ConfigManager(str(p))

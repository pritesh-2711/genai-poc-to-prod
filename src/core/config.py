"""Configuration management for the application."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

from .exceptions import ConfigurationError
from .models import ChatConfig, DBConfig, LLMConfig


class ConfigManager:
    """Manages application configuration loading and access."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """Initialize configuration manager.

        Args:
            config_path: Path to the configuration YAML file.

        Raises:
            ConfigurationError: If configuration file is not found or invalid.
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        self._load_env_variables()
        self.config = self._load_config()
        self.llm_config = self._build_llm_config()
        self.chat_config = self._build_chat_config()
        self.db_config = self._build_db_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the YAML configuration file.

        Returns:
            Loaded configuration dictionary.

        Raises:
            ConfigurationError: If YAML parsing fails.
        """
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
                if config is None:
                    raise ConfigurationError("Configuration file is empty")
                return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse configuration file: {e}")

    def _resolve_env_vars(self, value: Any) -> Any:
        """Resolve environment variables in configuration values.

        Args:
            value: The configuration value to process.

        Returns:
            The value with environment variables resolved.
        """
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, value)
        return value

    def _build_llm_config(self) -> LLMConfig:
        """Build LLM configuration from config file.

        Returns:
            LLMConfig object with provider-specific settings.

        Raises:
            ConfigurationError: If LLM configuration is invalid.
        """
        llm_config = self.config.get("llm", {})
        provider = llm_config.get("provider", "ollama").lower()

        if provider not in ["ollama", "openai"]:
            raise ConfigurationError(f"Unsupported LLM provider: {provider}")

        provider_config = llm_config.get(provider, {})

        return LLMConfig(
            provider=provider,
            model=provider_config.get("model", "mistral:7b" if provider == "ollama" else "gpt-4.1-nano"),
            temperature=provider_config.get("temperature", 0.7),
            max_tokens=provider_config.get("max_tokens"),
            api_key=self._resolve_env_vars(provider_config.get("api_key")),
            base_url=provider_config.get("base_url"),
        )

    def _build_chat_config(self) -> ChatConfig:
        """Build chat configuration from config file.

        Returns:
            ChatConfig object with chat-specific settings.
        """
        chat_config = self.config.get("chat", {})
        return ChatConfig(
            system_prompt=chat_config.get("system_prompt", "You are a helpful assistant."),
            timeout=chat_config.get("timeout", 30),
        )

    def _build_db_config(self) -> DBConfig:
        """Build database configuration from config file and environment variables.

        Returns:
            DBConfig object.

        Raises:
            ConfigurationError: If required DB config is missing.
        """
        db_config = self.config.get("database", {})

        host = self._resolve_env_vars(db_config.get("host", "${DB_HOST}"))
        port = int(self._resolve_env_vars(str(db_config.get("port", "${DB_PORT}"))))
        database = self._resolve_env_vars(db_config.get("database", "${DB_NAME}"))
        user = self._resolve_env_vars(db_config.get("user", "${DB_USER}"))
        password = self._resolve_env_vars(db_config.get("password", "${DB_PASSWORD}"))

        return DBConfig(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., "llm.ollama.model").
            default: Default value if key is not found.

        Returns:
            Configuration value.
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value

    def _load_env_variables(self) -> None:
        """Load environment variables for configuration."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            raise ConfigurationError(
                "python-dotenv is required. Install it with 'pip install python-dotenv'."
            )
        except Exception as e:
            raise ConfigurationError(f"Failed to load environment variables: {e}")
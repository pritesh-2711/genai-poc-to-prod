"""Logging configuration and setup."""

import logging
import logging.config
from pathlib import Path

import yaml

from .exceptions import ConfigurationError


class LoggingManager:
    """Manages application logging setup."""

    _initialized = False

    @classmethod
    def setup(cls, logging_config_path: str = "configs/logging.yaml") -> logging.Logger:
        """Initialize logging from YAML configuration.

        Args:
            logging_config_path: Path to the logging configuration file.

        Returns:
            The root logger instance.

        Raises:
            ConfigurationError: If configuration file is not found or invalid.
        """
        if cls._initialized:
            return logging.getLogger(__name__)

        config_path = Path(logging_config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Logging configuration file not found: {logging_config_path}")

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)

            logging.config.dictConfig(config)
            cls._initialized = True
            return logging.getLogger("research_paper_chat")

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse logging configuration: {e}")

    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance.

        Args:
            name: Logger name (typically __name__).

        Returns:
            A logger instance.
        """
        return logging.getLogger(name)

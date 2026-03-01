"""Base class and interface for LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..core.logging import LoggingManager
from ..core.models import ChatMessage
from .exceptions import LLMInitializationError

logger = LoggingManager.get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    Defines the interface that all LLM providers must implement.
    This follows the abstraction principle for easy extension to new providers.
    """

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        """Initialize the LLM provider.

        Args:
            model: Model identifier (e.g., "mistral:7b" or "gpt-4").
            temperature: Sampling temperature (0.0 to 1.0).
            **kwargs: Additional provider-specific arguments.

        Raises:
            LLMInitializationError: If initialization fails.
        """
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        logger.debug(f"Initializing {self.__class__.__name__} with model: {model}")

    @abstractmethod
    def chat(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Generate a response based on a user message.

        Args:
            user_message: The user's message.
            system_prompt: System prompt/instruction for the model.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response as a string.

        Raises:
            LLMProviderError: If generation fails.
        """
        pass

    @abstractmethod
    async def achat(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Asynchronously generate a response based on a user message.

        Args:
            user_message: The user's message.
            system_prompt: System prompt/instruction for the model.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response as a string.

        Raises:
            LLMProviderError: If generation fails.
        """
        pass

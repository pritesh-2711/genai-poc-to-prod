"""Base class and interface for LLM providers."""

from abc import ABC, abstractmethod

from ..core.logging import LoggingManager
from .exceptions import LLMInitializationError

logger = LoggingManager.get_logger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model: str, temperature: float = 0.7, **kwargs):
        self.model = model
        self.temperature = temperature
        self.kwargs = kwargs
        logger.debug(f"Initializing {self.__class__.__name__} with model: {model}")

    @abstractmethod
    def chat(self, user_message: str, system_prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def achat(self, user_message: str, system_prompt: str, **kwargs) -> str:
        pass
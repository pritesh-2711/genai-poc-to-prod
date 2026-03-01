"""LLM provider implementations."""

from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama as LangChainChatOllama
from langchain_openai import ChatOpenAI as LangChainChatOpenAI

from .core.logging import LoggingManager
from .core.models import ChatMessage
from .core.exceptions import LLMInitializationError, LLMProviderError
from .core.llm import BaseLLMProvider

logger = LoggingManager.get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """LLM provider using Ollama service.

    This provider uses the Ollama service for local LLM inference.
    """

    def __init__(
        self,
        model: str = "mistral:7b",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
        """Initialize Ollama provider.

        Args:
            model: Ollama model identifier.
            temperature: Sampling temperature.
            base_url: Base URL for Ollama service.
            **kwargs: Additional arguments.

        Raises:
            LLMInitializationError: If initialization fails.
        """
        super().__init__(model, temperature, **kwargs)
        self.base_url = base_url

        try:
            self.llm = LangChainChatOllama(
                model=model,
                base_url=base_url,
                temperature=temperature,
            )
            logger.info(f"Ollama provider initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama provider: {e}")
            raise LLMInitializationError(f"Ollama initialization failed: {e}")

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Generate a response using Ollama.

        Args:
            user_message: The user's message.
            system_prompt: System prompt for the model.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response string.

        Raises:
            LLMProviderError: If generation fails.
        """
        try:
            # Create prompt with system prompt and user message
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_message)
            ])

            # Create and execute chain
            chain = prompt | self.llm

            response = chain.invoke({})
            logger.debug("Generated response from Ollama")

            return response.content

        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise LLMProviderError(f"Ollama chat generation failed: {e}")

    async def achat(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Asynchronously generate a response using Ollama.

        Args:
            user_message: The user's message.
            system_prompt: System prompt for the model.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response string.

        Raises:
            LLMProviderError: If generation fails.
        """
        try:
            # Create prompt with system prompt and user message
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_message)
            ])

            # Create and execute chain
            chain = prompt | self.llm

            response = await chain.ainvoke({})
            logger.debug("Generated async response from Ollama")

            return response.content

        except Exception as e:
            logger.error(f"Ollama async chat failed: {e}")
            raise LLMProviderError(f"Ollama async chat generation failed: {e}")


class OpenAIProvider(BaseLLMProvider):
    """LLM provider using OpenAI service.

    This provider uses OpenAI's API for LLM inference.
    """

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Initialize OpenAI provider.

        Args:
            model: OpenAI model identifier.
            temperature: Sampling temperature.
            api_key: OpenAI API key.
            max_tokens: Maximum tokens in response.
            **kwargs: Additional arguments.

        Raises:
            LLMInitializationError: If initialization fails.
        """
        super().__init__(model, temperature, **kwargs)
        self.api_key = api_key
        self.max_tokens = max_tokens

        try:
            self.llm = LangChainChatOpenAI(
                model=model,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info(f"OpenAI provider initialized with model: {model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI provider: {e}")
            raise LLMInitializationError(f"OpenAI initialization failed: {e}")

    def chat(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Generate a response using OpenAI.

        Args:
            user_message: The user's message.
            system_prompt: System prompt for the model.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response string.

        Raises:
            LLMProviderError: If generation fails.
        """
        try:
            # Create prompt with system prompt and user message
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_message)
            ])

            # Create and execute chain
            chain = prompt | self.llm

            response = chain.invoke({})
            logger.debug("Generated response from OpenAI")

            return response.content

        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise LLMProviderError(f"OpenAI chat generation failed: {e}")

    async def achat(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs,
    ) -> str:
        """Asynchronously generate a response using OpenAI.

        Args:
            user_message: The user's message.
            system_prompt: System prompt for the model.
            **kwargs: Additional generation parameters.

        Returns:
            Generated response string.

        Raises:
            LLMProviderError: If generation fails.
        """
        try:
            # Create prompt with system prompt and user message
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_message)
            ])

            # Create and execute chain
            chain = prompt | self.llm

            response = await chain.ainvoke({})
            logger.debug("Generated async response from OpenAI")

            return response.content

        except Exception as e:
            logger.error(f"OpenAI async chat failed: {e}")
            raise LLMProviderError(f"OpenAI async chat generation failed: {e}")


class LLMProviderFactory:
    """Factory for creating LLM provider instances.

    This factory pattern allows easy selection of providers based on configuration.
    """

    _providers = {
        "ollama": OllamaProvider,
        "openai": OpenAIProvider,
    }

    @classmethod
    def create(
        cls,
        provider_name: str,
        model: str,
        temperature: float = 0.7,
        **kwargs,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.

        Args:
            provider_name: Name of the provider ("ollama" or "openai").
            model: Model identifier.
            temperature: Sampling temperature.
            **kwargs: Provider-specific arguments.

        Returns:
            An instance of the requested LLM provider.

        Raises:
            LLMProviderError: If provider is not supported.
            LLMInitializationError: If provider initialization fails.
        """
        provider_class = cls._providers.get(provider_name.lower())

        if not provider_class:
            supported = ", ".join(cls._providers.keys())
            raise LLMProviderError(
                f"Unsupported provider: {provider_name}. Supported providers: {supported}"
            )

        logger.debug(f"Creating {provider_name} provider with model: {model}")
        return provider_class(model=model, temperature=temperature, **kwargs)

    @classmethod
    def register_provider(cls, name: str, provider_class: type) -> None:
        """Register a new provider.

        This allows extension with custom providers while maintaining the abstraction.

        Args:
            name: Provider name.
            provider_class: Provider class (must inherit from BaseLLMProvider).
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"{provider_class} must inherit from BaseLLMProvider")

        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered custom provider: {name}")

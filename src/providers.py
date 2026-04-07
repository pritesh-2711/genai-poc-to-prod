"""LLM provider implementations."""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama as LangChainChatOllama
from langchain_openai import ChatOpenAI as LangChainChatOpenAI

from .core.logging import LoggingManager
from .core.exceptions import LLMInitializationError, LLMProviderError
from .core.llm import BaseLLMProvider

logger = LoggingManager.get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """LLM provider using Ollama service."""

    def __init__(
        self,
        model: str = "mistral:7b",
        temperature: float = 0.7,
        base_url: str = "http://localhost:11434",
        **kwargs,
    ):
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

    def chat(self, user_message: str, system_prompt: str, **kwargs) -> str:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            response = self.llm.invoke(messages)
            logger.debug("Generated response from Ollama")
            return response.content
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise LLMProviderError(f"Ollama chat generation failed: {e}")

    async def achat(self, user_message: str, system_prompt: str, **kwargs) -> str:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            response = await self.llm.ainvoke(messages)
            logger.debug("Generated async response from Ollama")
            return response.content
        except Exception as e:
            logger.error(f"Ollama async chat failed: {e}")
            raise LLMProviderError(f"Ollama async chat generation failed: {e}")

    async def astream_chat(self, user_message: str, system_prompt: str, **kwargs):
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Ollama stream chat failed: {e}")
            raise LLMProviderError(f"Ollama stream chat generation failed: {e}")


class OpenAIProvider(BaseLLMProvider):
    """LLM provider using OpenAI service."""

    def __init__(
        self,
        model: str = "gpt-4",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
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

    def chat(self, user_message: str, system_prompt: str, **kwargs) -> str:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            response = self.llm.invoke(messages)
            logger.debug("Generated response from OpenAI")
            return response.content
        except Exception as e:
            logger.error(f"OpenAI chat failed: {e}")
            raise LLMProviderError(f"OpenAI chat generation failed: {e}")

    async def achat(self, user_message: str, system_prompt: str, **kwargs) -> str:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            response = await self.llm.ainvoke(messages)
            logger.debug("Generated async response from OpenAI")
            return response.content
        except Exception as e:
            logger.error(f"OpenAI async chat failed: {e}")
            raise LLMProviderError(f"OpenAI async chat generation failed: {e}")

    async def astream_chat(self, user_message: str, system_prompt: str, **kwargs):
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"OpenAI stream chat failed: {e}")
            raise LLMProviderError(f"OpenAI stream chat generation failed: {e}")


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""

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
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"{provider_class} must inherit from BaseLLMProvider")
        cls._providers[name.lower()] = provider_class
        logger.info(f"Registered custom provider: {name}")
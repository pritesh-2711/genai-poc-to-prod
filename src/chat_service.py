"""Chat service for managing conversations with research papers."""

from typing import Optional

from .core.exceptions import ChatServiceError
from .core.logging import LoggingManager
from .core.models import ChatConfig, LLMConfig
from .providers import LLMProviderFactory, BaseLLMProvider

logger = LoggingManager.get_logger(__name__)


class ChatService:
    """Main service for managing chat conversations.

    This service orchestrates the interaction between the LLM provider,
    chat messages, and configuration.
    """

    def __init__(
        self,
        llm_config: LLMConfig,
        chat_config: ChatConfig,
    ):
        """Initialize the chat service.

        Args:
            llm_config: LLM configuration.
            chat_config: Chat configuration.

        Raises:
            ChatServiceError: If initialization fails.
        """
        try:
            self.llm_config = llm_config
            self.chat_config = chat_config

            # Initialize the LLM provider based on configuration
            self.llm_provider = self._initialize_provider()
            logger.info("ChatService initialized successfully")

        except Exception as e:
            logger.error(f"ChatService initialization failed: {e}")
            raise ChatServiceError(f"Failed to initialize ChatService: {e}")

    def _initialize_provider(self) -> BaseLLMProvider:
        """Initialize the LLM provider based on configuration.

        Returns:
            An initialized LLM provider instance.

        Raises:
            ChatServiceError: If provider initialization fails.
        """
        try:
            provider_kwargs = {}

            if self.llm_config.provider == "ollama":
                if self.llm_config.base_url:
                    provider_kwargs["base_url"] = self.llm_config.base_url

            elif self.llm_config.provider == "openai":
                if self.llm_config.api_key:
                    provider_kwargs["api_key"] = self.llm_config.api_key
                if self.llm_config.max_tokens:
                    provider_kwargs["max_tokens"] = self.llm_config.max_tokens

            provider = LLMProviderFactory.create(
                provider_name=self.llm_config.provider,
                model=self.llm_config.model,
                temperature=self.llm_config.temperature,
                **provider_kwargs,
            )

            return provider

        except Exception as e:
            logger.error(f"Provider initialization failed: {e}")
            raise ChatServiceError(f"Failed to initialize LLM provider: {e}")


    def get_response(self, user_message: str) -> str:
        """Get a response from the LLM for a user message.

        Args:
            user_message: The user's message.

        Returns:
            The LLM's response as a string.

        Raises:
            ChatServiceError: If response generation fails.
        """
        try:
            # Get response from LLM
            response = self.llm_provider.chat(
                user_message=user_message,
                system_prompt=self.chat_config.system_prompt,
            )

            logger.info("Successfully generated response")
            return response

        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            raise ChatServiceError(f"Failed to generate response: {e}")

    async def get_response_async(self, user_message: str) -> str:
        """Asynchronously get a response from the LLM.

        Args:
            user_message: The user's message.

        Returns:
            The LLM's response as a string.

        Raises:
            ChatServiceError: If response generation fails.
        """
        try:
            # Get response from LLM
            response = await self.llm_provider.achat(
                user_message=user_message,
                system_prompt=self.chat_config.system_prompt,
            )

            logger.info("Successfully generated async response")
            return response

        except Exception as e:
            logger.error(f"Failed to get async response: {e}")
            raise ChatServiceError(f"Failed to generate async response: {e}")

    def switch_provider(self, provider_name: str, model: Optional[str] = None) -> None:
        """Switch to a different LLM provider.

        Args:
            provider_name: Name of the new provider.
            model: Optional new model. If not provided, uses configured model.

        Raises:
            ChatServiceError: If provider switch fails.
        """
        try:
            old_provider = self.llm_config.provider
            self.llm_config.provider = provider_name

            if model:
                self.llm_config.model = model

            self.llm_provider = self._initialize_provider()
            logger.info(f"Switched provider from {old_provider} to {provider_name}")

        except Exception as e:
            logger.error(f"Failed to switch provider: {e}")
            raise ChatServiceError(f"Failed to switch provider: {e}")

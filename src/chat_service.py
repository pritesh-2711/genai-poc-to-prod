"""Chat service for managing conversations with research papers."""

from typing import List, Optional

from .core.exceptions import ChatServiceError
from .core.logging import LoggingManager
from .core.models import ChatConfig, ChatRecord, LLMConfig
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

    def get_response(
        self,
        user_message: str,
        history: Optional[List[ChatRecord]] = None,
    ) -> str:
        """Get a response from the LLM for a user message.

        Conversation history is injected into the system prompt so the model
        has full context of the current session.

        Args:
            user_message: The user's message.
            history: Prior chat records for the active session. Oldest first.

        Returns:
            The LLM's response as a string.

        Raises:
            ChatServiceError: If response generation fails.
        """
        try:
            response = self.llm_provider.chat(
                user_message=user_message,
                system_prompt=self._build_system_prompt(history),
            )
            logger.info("Successfully generated response")
            return response

        except Exception as e:
            logger.error(f"Failed to get response: {e}")
            raise ChatServiceError(f"Failed to generate response: {e}")

    async def get_response_async(
        self,
        user_message: str,
        history: Optional[List[ChatRecord]] = None,
    ) -> str:
        """Asynchronously get a response from the LLM.

        Args:
            user_message: The user's message.
            history: Prior chat records for the active session. Oldest first.

        Returns:
            The LLM's response as a string.

        Raises:
            ChatServiceError: If response generation fails.
        """
        try:
            response = await self.llm_provider.achat(
                user_message=user_message,
                system_prompt=self._build_system_prompt(history),
            )
            logger.info("Successfully generated async response")
            return response

        except Exception as e:
            logger.error(f"Failed to get async response: {e}")
            raise ChatServiceError(f"Failed to generate async response: {e}")

    def _build_system_prompt(self, history: Optional[List[ChatRecord]]) -> str:
        """Compose the system prompt, appending conversation history when available.

        Args:
            history: Ordered list of prior ChatRecord objects.

        Returns:
            Full system prompt string with history block appended if relevant.
        """
        base_prompt = self.chat_config.system_prompt

        if not history:
            return base_prompt

        history_lines = []
        for record in history:
            role_label = "User" if record.sender == "user" else "Assistant"
            history_lines.append(f"{role_label}: {record.message}")

        history_block = "\n".join(history_lines)

        return (
            f"{base_prompt}\n\n"
            f"The following is the conversation history for this session. "
            f"Use it to maintain context when responding.\n\n"
            f"--- Conversation History ---\n"
            f"{history_block}\n"
            f"--- End of History ---"
        )

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
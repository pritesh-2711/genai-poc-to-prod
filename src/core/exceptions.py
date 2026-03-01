"""Custom exceptions for the research paper chat application."""


class ResearchPaperChatException(Exception):
    """Base exception for the application."""

    pass


class ConfigurationError(ResearchPaperChatException):
    """Raised when there is a configuration error."""

    pass


class LLMProviderError(ResearchPaperChatException):
    """Raised when there is an issue with the LLM provider."""

    pass


class LLMInitializationError(LLMProviderError):
    """Raised when LLM initialization fails."""

    pass


class ChatServiceError(ResearchPaperChatException):
    """Raised when there is an issue with the chat service."""

    pass


class PromptValidationError(ResearchPaperChatException):
    """Raised when prompt validation fails."""

    pass

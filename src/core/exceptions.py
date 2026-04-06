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


class InputBlockedError(ResearchPaperChatException):
    """Raised when a user message is blocked by an input guardrail.

    Distinct from ChatServiceError so the API layer can return HTTP 400
    (bad request from the client) rather than HTTP 502 (LLM failure).
    """

    pass


class ExtractionError(ResearchPaperChatException):
    """Raised when document extraction fails (layout, text, table, or image)."""

    pass


class IngestionPipelineError(ResearchPaperChatException):
    """Raised when the extract → chunk → embed → ingest pipeline fails."""

    pass

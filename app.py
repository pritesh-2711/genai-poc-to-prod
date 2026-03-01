"""Chainlit UI for the research paper chat application."""

import chainlit as cl

from src.chat_service import ChatService
from src.core.config import ConfigManager
from src.core.logging import LoggingManager

# Setup logging
logger = LoggingManager.setup()

# Load configuration
config_manager = ConfigManager()

# Initialize chat service at startup
chat_service: ChatService = None


@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat when user starts a conversation."""
    global chat_service

    try:
        # Create chat service with configuration
        chat_service = ChatService(
            llm_config=config_manager.llm_config,
            chat_config=config_manager.chat_config,
        )

        # Display welcome message
        await cl.Message(
            content=f"Welcome to Research Paper Chat!\n\n"
            f"**Provider:** {config_manager.llm_config.provider.capitalize()}\n"
            f"**Model:** {config_manager.llm_config.model}\n\n"
            f"I'm here to help you discuss and analyze research papers. "
            f"Feel free to ask me anything about papers, methodologies, findings, or related topics."
        ).send()

        logger.info("Chat session started")

    except Exception as e:
        logger.error(f"Failed to initialize chat: {e}")
        await cl.Message(content=f"Error initializing chat: {str(e)}").send()


@cl.on_message
async def handle_message(message: cl.Message):
    """Handle incoming messages from the user."""
    global chat_service

    if chat_service is None:
        await cl.Message(content="Chat service not initialized. Please refresh and try again.").send()
        return

    try:
        # Get response from chat service
        response = await chat_service.get_response_async(message.content)

        # Send response back to user
        await cl.Message(content=response).send()

        logger.info("User message processed. Response sent.")

    except Exception as e:
        logger.error(f"Error processing message: {e}")
        await cl.Message(content=f"Error processing your message: {str(e)}").send()


@cl.on_chat_end
async def on_session_end():
    """Handle session end."""
    global chat_service

    if chat_service:
        chat_service = None

    logger.info("Chat session ended")

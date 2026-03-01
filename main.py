"""Main entry point for the research paper chat application."""

import argparse
import sys
from typing import Optional

from src.chat_service import ChatService
from src.core.config import ConfigManager
from src.core.logging import LoggingManager

# Setup logging
logger = LoggingManager.setup()


def main():
    """Main entry point for CLI mode."""
    parser = argparse.ArgumentParser(
        description="Research Paper Chat - Discuss research papers with an AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start interactive chat
  python main.py --web              # Start web UI with Chainlit
  python main.py --provider openai  # Use OpenAI provider
        """,
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Start web UI with Chainlit (requires: chainlit run app.py)",
    )
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai"],
        help="Override configured LLM provider",
    )
    parser.add_argument(
        "--model",
        help="Override configured model",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config_manager = ConfigManager(config_path=args.config)

        # Override provider if specified
        if args.provider:
            config_manager.llm_config.provider = args.provider

        # Override model if specified
        if args.model:
            config_manager.llm_config.model = args.model

        if args.web:
            logger.info("Starting web UI mode...")
            print("\n" + "=" * 60)
            print("To start the web UI, run:")
            print("  chainlit run app.py")
            print("=" * 60 + "\n")
            return 0

        # Start interactive CLI chat
        interactive_chat(config_manager)

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def interactive_chat(config_manager: ConfigManager) -> None:
    """Run interactive chat mode in the terminal.

    Args:
        config_manager: Configuration manager instance.
    """
    # Initialize chat service
    chat_service = ChatService(
        llm_config=config_manager.llm_config,
        chat_config=config_manager.chat_config,
    )

    # Display welcome message
    print("\n" + "=" * 60)
    print("Welcome to Research Paper Chat!")
    print("=" * 60)
    print(f"Provider: {config_manager.llm_config.provider}")
    print(f"Model: {config_manager.llm_config.model}")
    print(f"System Prompt: {config_manager.chat_config.system_prompt}")
    print("-" * 60)
    print("Commands:")
    print("  /quit    - Exit the chat")
    print("=" * 60 + "\n")

    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            # Get response from chat service
            logger.debug(f"User input: {user_input}")
            response = chat_service.get_response(user_input)

            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print(f"Error: {e}\n")


if __name__ == "__main__":
    sys.exit(main())

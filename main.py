"""Main entry point for the research paper chat application."""

import argparse
import getpass
import sys

from src.chat_service import ChatService
from src.core.config import ConfigManager
from src.core.logging import LoggingManager
from src.memory import MemoryRepository
from src.memory.repository import AuthenticationError

logger = LoggingManager.setup()


def main():
    parser = argparse.ArgumentParser(
        description="Research Paper Chat - Discuss research papers with an AI assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Start interactive chat
  python main.py --provider openai  # Use OpenAI provider
        """,
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
        config_manager = ConfigManager(config_path=args.config)

        if args.provider:
            config_manager.llm_config.provider = args.provider
        if args.model:
            config_manager.llm_config.model = args.model

        interactive_chat(config_manager)

    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def _login(repo: MemoryRepository):
    """Prompt for credentials and return an authenticated UserRecord.

    Retries up to 3 times before exiting.
    """
    print("\n" + "=" * 60)
    print("Login")
    print("=" * 60)

    for attempt in range(3):
        email = input("Email: ").strip()
        password = getpass.getpass("Password: ")
        try:
            user = repo.authenticate_user(email=email, password=password)
            print(f"Logged in as {user.name} ({user.email})\n")
            return user
        except AuthenticationError:
            remaining = 2 - attempt
            if remaining > 0:
                print(f"Invalid credentials. {remaining} attempt(s) remaining.\n")
            else:
                print("Too many failed attempts. Exiting.")
                sys.exit(1)


def _pick_or_create_session(repo: MemoryRepository, user_id):
    """List existing sessions and let the user pick one or create a new one."""
    sessions = repo.get_sessions(user_id)

    if not sessions:
        session = repo.create_session(user_id, "Session 1")
        print(f"No sessions found. Created new session: {session.session_name}")
        return session

    print("\nYour sessions:")
    for i, s in enumerate(sessions):
        status = "active" if s.is_active else "ended"
        ts = s.created_at.strftime("%b %d, %H:%M")
        print(f"  [{i + 1}] {s.session_name} ({status}, {ts})")
    print("  [N] Start a new session")

    while True:
        choice = input("\nPick a session [1-{}/N]: ".format(len(sessions))).strip().lower()
        if choice == "n":
            name = f"Session {len(sessions) + 1}"
            session = repo.create_session(user_id, name)
            print(f"Created new session: {session.session_name}")
            return session
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                selected = sessions[idx]
                print(f"Resuming session: {selected.session_name}")
                return selected
        print("Invalid choice. Try again.")


def interactive_chat(config_manager: ConfigManager) -> None:
    """Run interactive chat mode in the terminal."""
    repo = MemoryRepository(config_manager.db_config)
    user = _login(repo)
    session = _pick_or_create_session(repo, user.user_id)

    chat_service = ChatService(
        llm_config=config_manager.llm_config,
        chat_config=config_manager.chat_config,
    )

    print("\n" + "=" * 60)
    print("Research Paper Chat")
    print("=" * 60)
    print(f"Provider : {config_manager.llm_config.provider}")
    print(f"Model    : {config_manager.llm_config.model}")
    print(f"Session  : {session.session_name} ({session.session_id})")
    print("-" * 60)
    print("Commands: /quit  /history  /newsession  /endsession")
    print("=" * 60 + "\n")

    # Replay history for the chosen session
    history = repo.get_conversation_history(session.session_id)
    if history:
        print(f"--- {len(history)} previous message(s) in this session ---")
        for record in history:
            label = "You" if record.sender == "user" else "Assistant"
            print(f"{label}: {record.message}\n")
        print("--- End of history ---\n")

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() == "/quit":
                print("Goodbye!")
                break

            if user_input.lower() == "/history":
                records = repo.get_conversation_history(session.session_id)
                if not records:
                    print("No messages in this session yet.\n")
                else:
                    for record in records:
                        label = "You" if record.sender == "user" else "Assistant"
                        print(f"{label}: {record.message}\n")
                continue

            if user_input.lower() == "/newsession":
                sessions = repo.get_sessions(user.user_id)
                name = f"Session {len(sessions) + 1}"
                session = repo.create_session(user.user_id, name)
                print(f"Started new session: {session.session_name} ({session.session_id})\n")
                continue

            if user_input.lower() == "/endsession":
                repo.terminate_session(session.session_id)
                sessions = repo.get_sessions(user.user_id)
                name = f"Session {len(sessions) + 1}"
                session = repo.create_session(user.user_id, name)
                print(f"Session ended. Started new session: {session.session_name}\n")
                continue

            # Persist user message
            repo.add_message(session.session_id, "user", user_input)

            # Build context from history (exclude the message just added)
            full_history = repo.get_conversation_history(session.session_id)
            context_history = full_history[:-1] if full_history else []

            response = chat_service.get_response(
                user_message=user_input,
                history=context_history,
            )

            repo.add_message(session.session_id, "assistant", response)
            print(f"\nAssistant: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}")
            print(f"Error: {e}\n")


if __name__ == "__main__":
    sys.exit(main())
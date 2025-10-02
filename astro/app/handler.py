# --- External Imports ---
from pathlib import Path

from langchain_core.messages import BaseMessage
from pydantic import Field
from sqlmodel import Session, select

# --- Local Imports ---
from astro.agents.base import Agent, AgentConfig
from astro.llms.base import LLMConfig
from astro.loggings import get_loggy
from astro.paths import ModelFileStore, get_stores_dir, setup_paths
from astro.stores import Store, get_store
from astro.typings import ModelName, ModelProvider, RecordableModel

# Global logger
_logger = get_loggy(__file__)


class Chat(RecordableModel, frozen=True):
    """Immutable chat conversation record.

    Stores the complete state of a chat conversation including
    the agent configuration and all messages.
    """

    agent_config: AgentConfig
    messages: list[BaseMessage] = Field(default_factory=list)


class ChatHandler:
    """Handles chat management with two-tier storage system.

    Manages chat conversations using:
    - Tier 1: Local file system (ModelFileStore) for fast access
    - Tier 2: SQLite database (Store) for persistent async backup

    The handler maintains the current active chat and agent, allowing
    users to switch between chats and models seamlessly.
    """

    def __init__(self, db_engine=None) -> None:
        """Initialize the ChatHandler with storage systems.

        Args:
            db_engine: Optional database engine for Store system
        """
        _logger.info("Initializing ChatHandler")

        # Setup paths if not already done
        setup_paths()

        # Current state
        self._current_chat: Chat | None = None
        self._current_agent: Agent | None = None

        # Tier 1: Fast local file storage
        stores_dir = get_stores_dir()
        self._chat_file_store = ModelFileStore(
            root_dir=stores_dir / "chats", model_type=Chat
        )
        self._agent_config_file_store = ModelFileStore(
            root_dir=stores_dir / "agent_configs", model_type=AgentConfig
        )
        self._llm_config_file_store = ModelFileStore(
            root_dir=stores_dir / "llm_configs", model_type=LLMConfig
        )

        # Tier 2: Persistent database storage (async)
        # These will be registered and managed by the store system
        self._db_engine = db_engine

        _logger.info("ChatHandler initialized successfully")

    def create_new_chat(
        self,
        identifier: str | ModelName | ModelProvider,
        provider: str | ModelProvider | None = None,
    ) -> Agent:
        """Create a new chat with the specified model.

        Args:
            identifier: Model name or provider identifier
            provider: Optional model provider (inferred if not provided)

        Returns:
            Agent: The newly created chat agent
        """
        _logger.info(
            f"Creating new chat with identifier={identifier}, provider={provider}"
        )

        # Create agent config
        agent_config = AgentConfig.for_chat(identifier=identifier, provider=provider)

        # Create agent from config
        agent = Agent(config=agent_config)

        # Create initial chat record
        chat = Chat(
            agent_config=agent_config,
            messages=agent.messages,
        )

        # Save to both storage tiers
        self._save_chat(chat)

        # Set as current
        self._current_chat = chat
        self._current_agent = agent

        _logger.info(f"Created new chat with uid={chat.uid}")
        return agent

    def switch_chat(self, chat_uid: str) -> Agent:
        """Switch to an existing chat by its UID.

        Args:
            chat_uid: Unique identifier of the chat to switch to

        Returns:
            Agent: The agent for the switched chat

        Raises:
            KeyError: If chat_uid is not found
        """
        _logger.info(f"Switching to chat with uid={chat_uid}")

        # Load chat from file store (fast)
        chat = self._chat_file_store.get_model(chat_uid)

        if chat is None:
            _logger.error(f"Chat with uid={chat_uid} not found")
            raise KeyError(f"Chat with uid={chat_uid} not found")

        # Create agent from loaded chat
        agent = Agent(config=chat.agent_config)

        # Restore messages
        agent._messages = list(chat.messages)

        # Set as current
        self._current_chat = chat
        self._current_agent = agent

        _logger.info(f"Switched to chat uid={chat_uid}")
        return agent

    def update_current_chat(self) -> None:
        """Update the current chat with latest messages from agent.

        Should be called after agent interactions to persist changes.
        """
        if self._current_agent is None or self._current_chat is None:
            _logger.warning("No current chat to update")
            return

        _logger.debug("Updating current chat with latest messages")

        # Create new chat with updated messages (immutable)
        updated_chat = Chat(
            agent_config=self._current_chat.agent_config,
            messages=self._current_agent.messages,
        )

        # Save updated version
        self._save_chat(updated_chat)

        # Update current reference
        self._current_chat = updated_chat

        _logger.debug(f"Updated chat uid={updated_chat.uid}")

    def list_chats(self) -> list[str]:
        """List all available chat UIDs.

        Returns:
            List of chat UIDs
        """
        return list(self._chat_file_store.keys())

    def delete_chat(self, chat_uid: str) -> None:
        """Delete a chat from both storage tiers.

        Args:
            chat_uid: Unique identifier of the chat to delete
        """
        _logger.info(f"Deleting chat uid={chat_uid}")

        # Remove from file store
        if chat_uid in self._chat_file_store:
            self._chat_file_store.remove_model(chat_uid)

        # Clear current if it's the deleted chat
        if self._current_chat and self._current_chat.uid == chat_uid:
            self._current_chat = None
            self._current_agent = None

        _logger.info(f"Deleted chat uid={chat_uid}")

    def _save_chat(self, chat: Chat) -> None:
        """Save chat to both storage tiers with proper foreign key handling.

        Args:
            chat: Chat instance to save
        """
        _logger.debug(f"Saving chat uid={chat.uid}")

        # Tier 1: Save to file store (immediate, flat JSON)
        self._chat_file_store.add_model(chat)

        # Also save dependencies to file stores
        self._agent_config_file_store.add_model(chat.agent_config)
        self._llm_config_file_store.add_model(chat.agent_config.llm_config)

        # Tier 2: Save to database with foreign keys (async via Store)
        if self._db_engine is not None:
            self._save_to_database(chat)

        _logger.debug(f"Saved chat uid={chat.uid} to both storage tiers")

    def _save_to_database(self, chat: Chat) -> None:
        """Save chat to database with proper cascading foreign key relationships.

        This method handles the normalized database structure by:
        1. Ensuring LLMConfig exists and getting its uid
        2. Ensuring AgentConfig exists with FK to LLMConfig and getting its uid
        3. Saving Chat with FK to AgentConfig

        Args:
            chat: Chat instance to save to database
        """
        _logger.debug(f"Saving chat uid={chat.uid} to database")

        with Session(self._db_engine) as session:
            # Step 1: Handle LLMConfig (bottom of dependency chain)
            llm_config = chat.agent_config.llm_config
            llm_config_hash = hash(llm_config)

            # Check if LLMConfig already exists by hash
            llm_record = session.exec(
                select(LLMConfigRecord).where(
                    LLMConfigRecord.record_hash == llm_config_hash
                )
            ).first()

            if llm_record is None:
                # Create new LLMConfig record
                llm_record = LLMConfigRecord.from_model(llm_config)
                session.add(llm_record)
                session.commit()
                session.refresh(llm_record)
                _logger.debug(f"Created new LLMConfigRecord with uid={llm_record.uid}")
            else:
                _logger.debug(
                    f"Using existing LLMConfigRecord with uid={llm_record.uid}"
                )

            # Step 2: Handle AgentConfig (middle of dependency chain)
            agent_config = chat.agent_config
            agent_config_hash = hash(agent_config)

            # Check if AgentConfig already exists by hash
            agent_record = session.exec(
                select(AgentConfigRecord).where(
                    AgentConfigRecord.record_hash == agent_config_hash
                )
            ).first()

            if agent_record is None:
                # Create new AgentConfig record with FK to LLMConfig
                agent_record = AgentConfigRecord.from_model(
                    agent_config, llm_config_record_uid=llm_record.uid
                )
                session.add(agent_record)
                session.commit()
                session.refresh(agent_record)
                _logger.debug(
                    f"Created new AgentConfigRecord with uid={agent_record.uid}"
                )
            else:
                _logger.debug(
                    f"Using existing AgentConfigRecord with uid={agent_record.uid}"
                )

            # Step 3: Handle Chat (top of dependency chain)
            chat_hash = hash(chat)

            # Check if Chat already exists by hash
            chat_record = session.exec(
                select(ChatRecord).where(ChatRecord.record_hash == chat_hash)
            ).first()

            if chat_record is None:
                # Create new Chat record with FK to AgentConfig
                chat_record = ChatRecord.from_model(
                    chat, agent_config_record_uid=agent_record.uid
                )
                session.add(chat_record)
                session.commit()
                _logger.debug(f"Created new ChatRecord with uid={chat_record.uid}")
            else:
                _logger.debug(
                    f"Chat already exists in database with uid={chat_record.uid}"
                )

    def _load_from_database(self, chat_uid: str) -> Chat | None:
        """Load chat from database with proper foreign key resolution.

        This method reconstructs the full Chat object by:
        1. Loading ChatRecord
        2. Loading referenced AgentConfigRecord via FK
        3. Loading referenced LLMConfigRecord via FK
        4. Reconstructing nested Chat -> AgentConfig -> LLMConfig structure

        Args:
            chat_uid: UID of the chat to load

        Returns:
            Reconstructed Chat object or None if not found
        """
        if self._db_engine is None:
            return None

        _logger.debug(f"Loading chat uid={chat_uid} from database")

        with Session(self._db_engine) as session:
            # Find chat by hash (stored as record_hash)
            # Note: This assumes chat_uid is the hex representation of the hash
            try:
                chat_hash = int(chat_uid, 16)
            except ValueError:
                _logger.error(f"Invalid chat_uid format: {chat_uid}")
                return None

            chat_record = session.exec(
                select(ChatRecord).where(ChatRecord.record_hash == chat_hash)
            ).first()

            if chat_record is None:
                _logger.debug(f"Chat uid={chat_uid} not found in database")
                return None

            # Load AgentConfig via foreign key
            agent_record = session.get(AgentConfigRecord, chat_record.agent_config_uid)
            if agent_record is None:
                _logger.error(
                    f"AgentConfigRecord uid={chat_record.agent_config_uid} not found"
                )
                return None

            # Load LLMConfig via foreign key
            llm_record = session.get(LLMConfigRecord, agent_record.llm_config_uid)
            if llm_record is None:
                _logger.error(
                    f"LLMConfigRecord uid={agent_record.llm_config_uid} not found"
                )
                return None

            # Reconstruct from bottom up
            llm_config = llm_record.to_model()
            agent_config = agent_record.to_model(llm_config)
            chat = chat_record.to_model(agent_config)

            _logger.debug(f"Loaded chat uid={chat_uid} from database")
            return chat

    @property
    def current_agent(self) -> Agent | None:
        """Get the current active agent."""
        return self._current_agent

    @property
    def current_chat(self) -> Chat | None:
        """Get the current active chat."""
        return self._current_chat

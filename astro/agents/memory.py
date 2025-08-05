from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field

from astro.agents.base import MemoryModule, StrMessage
from astro.agents.effect import LLMInput, LLMMessage, LLMOutput
from astro.utilities.display import get_terminal_width


class ChatMemory(MemoryModule[int, LLMMessage]):
    """Memory for storing chat messages in a conversation.

    This class provides methods to add messages of different roles (user, AI, system, tool)
    and retrieve the last message or print the conversation in a formatted way.
    """

    def __init__(self, values: Optional[list[LLMMessage]] = None):
        self._messages = values or []

    @property
    def last(self) -> LLMMessage:
        """Return last message"""
        if len(self._messages) == 0:
            raise IndexError("No messages available")
        return self._messages[-1]

    @property
    def values(self) -> list[LLMMessage]:
        return self._messages

    def add(self, value: LLMMessage):
        """Add LangChain message directly."""
        self._messages.append(value)

    def add_ai_message(self, message: StrMessage):
        """Add AI message to chat memory"""
        self.add(LLMMessage(content=message, role="ai"))

    def add_human_message(self, message: StrMessage):
        """Add human message to chat memory"""
        self.add(LLMMessage(content=message, role="human"))

    def add_system_message(self, message: StrMessage):
        """Add system message to chat memory"""
        self.add(LLMMessage(content=message, role="system"))

    def add_tool_message(self, message: StrMessage, call_id: Optional[str] = None):
        """Add human message to chat memory"""
        if call_id:
            self.add(LLMMessage(uid=call_id, content=message, role="tool"))
        else:
            self.add(LLMMessage(content=message, role="tool"))

    def clear(self):
        """Clear messages in message memory"""
        self._messages = []

    def filter_by(self, role: str) -> list[LLMMessage]:
        """Filter messages by role."""
        if role not in ["human", "ai", "system", "tool"]:
            raise ValueError("Role must be one of: human, ai, system, tool")

        return [message for message in self._messages if message.role == role]

    def from_llm_output(self, outputs: LLMOutput):
        self.add(outputs.message)

    def to_llm_input(self) -> LLMInput:
        return LLMInput(messages=self._messages)

    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(key)

    def __len__(self) -> int:
        """Return the number of messages in memory."""
        return len(self._messages)

    def __repr__(self) -> str:
        """Return a formatted string representation of the conversation."""
        ai_count = len(self.filter_by("ai"))
        human_count = len(self.filter_by("human"))
        system_count = len(self.filter_by("system"))
        tool_count = len(self.filter_by("tool"))
        total_count = len(self._messages)
        digits = len(str(total_count))
        return (
            "ChatMemory(\n"
            + f"  # Total messages: {total_count}, AI: {ai_count}, User: {human_count}, System: {system_count}, Tool: {tool_count}\n"
            + "\n".join(
                f"  {i:<{digits}} -> {message!r}"
                for i, message in enumerate(self._messages)
            )
            + "\n)"
        )

    def __str__(self) -> str:
        """
        Return a formatted string representation of the conversation.

        Each message is displayed on its own line, prefixed by the sender's role in uppercase
        and aligned for readability.

        Returns:
            str: The formatted conversation history.

        Example:
            SYSTEM : You are a helpful assistant.
            HUMAN  : What is the capital of France?
            AI     : The capital of France is Paris.
            TOOL   : Weather in Paris: 18°C, partly cloudy

        Raises:
            None

        Note:
            If a message's content is a list or dict, it will be stringified.
        """
        return "\n".join(f"{message}" for message in self._messages)


class WorkMemory(BaseModel):
    """Memory for storing key-value pairs in a conversation.

    This class provides methods to read, write, and clear key-value pairs.
    It can be used for short-term or long-term memory storage.
    """

    memory: dict[str, Any] = Field(
        default_factory=dict, description="Dictionary to store key-value pairs"
    )

    def __len__(self) -> int:
        """Return the number of key-value pairs in memory."""
        return len(self.memory)

    def read(self, key: str) -> Any:
        """Read value from memory by key."""
        if key not in self.memory:
            raise KeyError(f"Key '{key}' not found in memory")
        return self.memory[key]

    def write(self, key: str, value: Any):
        """Write value to memory with specified key."""
        self.memory[key] = value

    def clear(self):
        """Clear all key-value pairs in memory."""
        self.memory.clear()


class ShortMemory(BaseModel):
    @abstractmethod
    def get(self, key: str) -> Any: ...

    @abstractmethod
    def set(self, key: str, value: Any): ...

    @abstractmethod
    def evict(self, key: str): ...


class LongTermMemory(ABC):
    @abstractmethod
    def query(self, query: str) -> list[Any]: ...

    @abstractmethod
    def store(self, record: Any): ...


if __name__ == "__main__":
    # Create a MessageMemory instance
    memory = ChatMemory()

    # Test basic functionality
    assert len(memory) == 0
    print("✓ Empty memory initialized")

    # Add system message
    memory.add_system_message("You are a helpful assistant.")
    assert len(memory) == 1
    print("✓ System message added")

    # Simulate a multi-turn conversation
    conversation_turns = [
        ("ai", "How can I help?"),
        ("human", "What is the capital of France?"),
        ("ai", "The capital of France is Paris."),
        ("human", "What about Italy?"),
        ("ai", "The capital of Italy is Rome."),
        ("human", "Tell me about these cities."),
        ("ai", "Paris is known for the Eiffel Tower and Rome for the Colosseum."),
        ("tool", "Weather in Paris: 18°C, partly cloudy"),
        ("ai", "The current weather in Paris is 18°C and partly cloudy."),
    ]

    # Add conversation messages
    for role, message in conversation_turns:
        if role == "human":
            memory.add_human_message(message)
        elif role == "ai":
            memory.add_ai_message(message)
        elif role == "tool":
            memory.add_tool_message(message, call_id="test")
        elif role == "system":
            memory.add_system_message(message)

    print(f"✓ Added {len(conversation_turns)} conversation messages")
    assert len(memory) == 10  # 1 system + 9 conversation messages

    # Test last message property
    last_message = memory.last
    assert last_message == "The current weather in Paris is 18°C and partly cloudy."
    print("✓ Last message retrieved correctly")

    # Print the full conversation
    print("\n" + "=" * get_terminal_width())
    print(memory)
    print("=" * get_terminal_width())
    print(repr(memory))
    print("=" * get_terminal_width())
    print(memory.values)
    print("=" * get_terminal_width())
    print(memory.to_llm_input())
    print("=" * get_terminal_width())
    print(memory.to_llm_input().to_langchain())
    print("=" * get_terminal_width())

    # Test edge cases
    empty_memory = ChatMemory()
    try:
        _ = empty_memory.last
        assert False, "Should have raised IndexError"
    except IndexError:
        print("✓ IndexError correctly raised for empty memory")

    print(f"\n✓ All tests passed! Memory contains {len(memory)} messages")
    print()

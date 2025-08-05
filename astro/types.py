"""
astro/typing.py

Core type definitions and base agent abstractions.

Author: Your Name
Date: 2025-07-24
License: MIT

Description:
    Provides protocols, type aliases, and abstract base classes for agent state and behavior.

Dependencies:
    - pydantic
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Protocol, TypeAlias

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


# ============================================================================
# Core Protocols and Type Aliases
# ============================================================================


class StringableProtocol(Protocol):
    """Protocol for objects convertible to strings."""

    def __str__(self) -> str: ...


# Unified Message Type System
MessageDictType: TypeAlias = dict[str, Any]
LangChainMessageType: TypeAlias = BaseMessage | HumanMessage | AIMessage | SystemMessage | ToolMessage

# Core message type - supports both dict and LangChain formats
MessageType: TypeAlias = str | MessageDictType | LangChainMessageType

# Sequence types
MessageSequenceType: TypeAlias = list[MessageType]

# Input types - single message or sequence
MessageInputType: TypeAlias = MessageType | MessageSequenceType

# ============================================================================
# I/O Schema Types (LangGraph-Compatible)
# ============================================================================

class AgentInput(BaseModel):
    """Standard input schema following LangChain patterns"""

class AgentOutput(BaseModel):
    """Standard output schema following LangChain patterns"""

class ChatInput(AgentInput):
    """Standard user chat message input for chat agents"""
    message: MessageInputType = Field(..., description="User message")

class ChatOutput(AgentOutput):
    """Standard agent chat message output for chat agents"""
    message: MessageType = Field(..., description="Agent response message")

class StructuredInput(AgentInput):
    """Standrd input for structured-based inputs for agents"""

class StructuredOutput(AgentOutput):
    """Standard output for structured-based outputs for agents"""

# ============================================================================
# State Types (LangGraph-Compatible)
# ============================================================================


class BaseState(ABC, BaseModel):
    """Base abstract class for agent states using Pydantic v2"""

    pass

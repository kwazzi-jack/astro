"""Agent output models and conversion helpers for Astro."""

# -- Internal Imports ---
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Literal, TypeAlias, TypeVar

# --- External Imports ---
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from pydantic_ai import (
    AgentRunResult,
    ModelResponsePart,
    ModelResponsePartDelta,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)

# --- Local Imports ---
from astro.typings.base import ArgsAdapter, NamedDict, type_name
from astro.utilities.uid import generate_double_uid

# --- Output Type Aliases ---
OutputType = TypeVar("OutputType")
AgentOutputKind = Literal[
    "text", "think", "tool_call", "tool_return", "final", "unknown"
]
AgentOutputType: TypeAlias = (
    "AgentText | AgentThink | AgentToolCall | AgentToolReturn | AgentFinal"
)
StreamOut = AsyncIterator["AgentOutput"]


# --- Agent Output Classes ---
class AgentOutput(BaseModel):
    """Base model representing a streamed chunk from an agent."""

    agent_name: str
    uid: str | None = None
    kind: AgentOutputKind = "unknown"
    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def generate_uid_if_needed(cls, data: Any) -> Any:
        """Generate UID before validation if not provided."""

        if isinstance(data, dict):
            uid = data.get("uid")
            if uid is None or (isinstance(uid, str) and len(uid) == 0):
                agent_name = data.get("agent_name", "agent")
                kind_field = cls.model_fields.get("kind")
                kind_value = data.get(
                    "kind",
                    kind_field.get_default(call_default_factory=True)
                    if kind_field
                    else "unknown",
                )
                data["uid"] = generate_double_uid(agent_name, kind_value)
        return data


class AgentText(AgentOutput):
    """Text content streamed from the agent."""

    text: str
    kind: AgentOutputKind = "text"

    @classmethod
    def from_part(cls, agent_name: str, part: TextPart) -> "AgentText":
        if isinstance(part, TextPart):
            return cls(agent_name=agent_name, text=part.content)
        raise ValidationError(
            f"Invalid input type: Expected TextPart, got {type_name(part)}."
        )

    @classmethod
    def from_delta(cls, agent_name: str, part_delta: TextPartDelta) -> "AgentText":
        if isinstance(part_delta, TextPartDelta):
            return cls(agent_name=agent_name, text=part_delta.content_delta)
        raise ValidationError(
            f"Invalid input type: Expected TextPartDelta, got {type_name(part_delta)}."
        )


class AgentThink(AgentOutput):
    """Thinking content streamed from the agent."""

    text: str | None
    signature: str | None = None
    kind: AgentOutputKind = "think"

    @classmethod
    def from_part(cls, agent_name: str, part: ThinkingPart) -> "AgentThink":
        if isinstance(part, ThinkingPart):
            return cls(agent_name=agent_name, text=part.content)
        raise ValidationError(
            f"Invalid input type: Expected ThinkingPart, got {type_name(part)}."
        )

    @classmethod
    def from_delta(cls, agent_name: str, part_delta: ThinkingPartDelta) -> "AgentThink":
        if isinstance(part_delta, ThinkingPartDelta):
            return cls(
                agent_name=agent_name,
                text=part_delta.content_delta,
                signature=part_delta.signature_delta,
            )
        raise ValidationError(
            "Invalid input type: Expected ThinkingPartDelta, "
            f"got {type_name(part_delta)}."
        )


class AgentToolCall(AgentOutput):
    """Tool call event emitted by the agent."""

    name: str | None = None
    arguments: NamedDict | None = None
    call_id: str | None = None
    kind: AgentOutputKind = "tool_call"

    @classmethod
    def from_part(cls, agent_name: str, part: ToolCallPart) -> "AgentToolCall":
        if isinstance(part, ToolCallPart):
            return cls(
                agent_name=agent_name,
                name=part.tool_name,
                arguments=part.args_as_dict(),
                call_id=part.tool_call_id,
            )
        raise ValidationError(
            f"Invalid input type: Expected ToolCallPart, got {type_name(part)}."
        )

    @classmethod
    def from_delta(
        cls, agent_name: str, part_delta: ToolCallPartDelta
    ) -> "AgentToolCall":
        if isinstance(part_delta, ToolCallPartDelta):
            raw_args = part_delta.args_delta
            if isinstance(raw_args, str):
                decoded_args = ArgsAdapter.validate_json(raw_args)
            elif raw_args is None:
                decoded_args = None
            else:
                decoded_args = raw_args
            return cls(
                agent_name=agent_name,
                name=part_delta.tool_name_delta,
                arguments=decoded_args,
                call_id=part_delta.tool_call_id,
            )
        raise ValidationError(
            "Invalid input type: Expected ToolCallPartDelta, "
            f"got {type_name(part_delta)}."
        )


class AgentToolReturn(AgentOutput):
    """Tool return event emitted by the agent."""

    name: str | None = None
    content: Any | None = None
    call_id: str | None = None
    kind: AgentOutputKind = "tool_return"

    @classmethod
    def from_part(cls, agent_name: str, part: ToolReturnPart) -> "AgentToolReturn":
        if isinstance(part, ToolReturnPart):
            return cls(
                agent_name=agent_name,
                name=part.tool_name,
                content=part.content,
                call_id=part.tool_call_id,
            )
        raise ValidationError(
            f"Invalid input type: Expected ToolReturnPart, got {type_name(part)}."
        )


class AgentFinal[OutputType](AgentOutput):
    """Final result event emitted by the agent.

    Attributes:
        output (OutputType | None): Final output payload produced by the agent.
        timestamp (datetime | None): Timestamp for the final agent response when available.
    """

    output: OutputType | None = None
    kind: AgentOutputKind = "final"

    @classmethod
    def from_run_result(
        cls, agent_name: str, run_result: AgentRunResult[OutputType]
    ) -> "AgentFinal[OutputType]":
        """Create a final output event from a completed agent run.

        Args:
            agent_name (str): Name of the agent emitting the final result.
            run_result (AgentRunResult[OutputType]): Completed agent run result instance.

        Returns:
            AgentFinal[OutputType]: Final output event carrying the aggregated result data.
        """
        return cls(agent_name=agent_name, output=run_result.output)


def part_to_agent_output(agent_name: str, part: ModelResponsePart) -> AgentOutput:
    """Convert a pydantic-ai response part into a unified agent output."""

    if isinstance(part, TextPart):
        return AgentText.from_part(agent_name, part)
    if isinstance(part, ThinkingPart):
        return AgentThink.from_part(agent_name, part)
    if isinstance(part, ToolCallPart):
        return AgentToolCall.from_part(agent_name, part)
    if isinstance(part, ToolReturnPart):
        return AgentToolReturn.from_part(agent_name, part)
    raise ValueError(f"Unknown pydantic-ai part-type {type_name(part)}")


def delta_to_agent_output(agent_name: str, part: ModelResponsePartDelta) -> AgentOutput:
    """Convert a pydantic-ai response delta into a unified agent output."""

    if isinstance(part, TextPartDelta):
        return AgentText.from_delta(agent_name, part)
    if isinstance(part, ThinkingPartDelta):
        return AgentThink.from_delta(agent_name, part)
    if isinstance(part, ToolCallPartDelta):
        return AgentToolCall.from_delta(agent_name, part)
    raise ValueError(f"Unknown pydantic-ai delta-type {type_name(part)}")


__all__ = [
    "OutputType",
    "AgentOutput",
    "AgentToolReturn",
    "AgentToolCall",
    "AgentThink",
    "AgentText",
    "AgentFinal",
    "AgentOutputKind",
    "AgentOutputType",
    "StreamOut",
    "delta_to_agent_output",
    "part_to_agent_output",
]

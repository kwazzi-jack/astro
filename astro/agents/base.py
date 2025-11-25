# --- Internal Imports ---
from collections.abc import Sequence
from types import NoneType
from typing import cast

# --- External Imports ---
from pydantic_ai import (
    Agent,
    AgentRunResult,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelSettings,
    PartDeltaEvent,
    PartStartEvent,
    ToolReturnPart,
)
from pydantic_ai.models import Model

# --- Local Imports ---
from astro.llms import create_llm_model
from astro.logger import get_loggy
from astro.typings.base import MessageList, type_name
from astro.typings.callables import AgentFn, AgentFnSequence, StreamFn
from astro.typings.contexts import ContextType
from astro.typings.inputs import StreamIn
from astro.typings.outputs import (
    AgentFinal,
    AgentText,
    AgentThink,
    AgentToolCall,
    AgentToolReturn,
    OutputType,
    StreamOut,
)

# --- Globals ---
_loggy = get_loggy(__file__)


# --- Agent Factory ---
def create_agent(
    identifier: str,
    instructions: str | Sequence[str] | None = None,
    context_type: type[ContextType] | None = None,
    tools: AgentFn[ContextType] | AgentFnSequence[ContextType] | None = None,
    output_type: type[OutputType] = str,
    model_settings: ModelSettings | None = None,
    agent_name: str = "agent",
) -> Agent[ContextType, OutputType]:
    deps_type: type[ContextType]
    if context_type is None:
        deps_type = cast("type[ContextType]", NoneType)
    else:
        deps_type = context_type

    if tools is None:
        resolved_tools: Sequence[AgentFn[ContextType]] = ()
    elif isinstance(tools, Sequence) and not isinstance(tools, (str, bytes)):
        resolved_tools = tuple(tools)
    else:
        resolved_tools = (cast("AgentFn[ContextType]", tools),)

    resolved_output_type = cast("type[OutputType]", output_type or str)
    try:
        model = create_llm_model(identifier)
    except Exception as error:
        raise _loggy.CreationError(object_type=Model, caused_by=error)
    return Agent(
        name=agent_name,
        model=model,
        instructions=instructions,
        deps_type=deps_type,
        model_settings=model_settings,
        tools=resolved_tools,
        output_type=resolved_output_type,
    )


# --- Stream Functions ---
def create_agent_stream(
    agent: Agent[ContextType, OutputType], context: ContextType, messages: MessageList
) -> StreamFn:
    """Create a streaming wrapper that yields unified agent outputs.

    Args:
        agent (Agent[ContextType, OutputType]): Agent instance to wrap.
        context (ContextType): Dependency context shared across runs.

    Returns:
        AgentStream: Async callable streaming AgentOutput items.
    """

    async def agent_stream(stream_input: StreamIn) -> StreamOut:
        """Stream normalised agent output parts for a prompt.

        Args:
            stream_input (InputType): User supplied input type.

        Yields:
            AgentOutput: Structured output chunk produced by the agent.

        Raises:
            ExpectedTypeError: Raised when prompt is not a string instance.
        """

        # Get agent name
        agent_name = agent.name if agent.name is not None else "agent"

        # Return result based on agent output type
        run_result: AgentRunResult[OutputType] | None = None

        # Run stream
        try:
            async with agent.iter(
                stream_input,
                deps=context,
                message_history=messages,
            ) as agent_run:
                _loggy.debug(f"Streaming {agent_name} run", prompt=stream_input)

                # Iterate over agent nodes
                async for node in agent_run:
                    # Model request
                    if Agent.is_model_request_node(node):
                        async with node.stream(agent_run.ctx) as model_stream:
                            async for event in model_stream:
                                # Start of part
                                if isinstance(event, PartStartEvent):
                                    part = event.part
                                    if part.part_kind == "text":
                                        output = AgentText.from_part(agent_name, part)
                                    elif part.part_kind == "thinking":
                                        output = AgentThink.from_part(agent_name, part)
                                    elif part.part_kind == "tool-call":
                                        output = AgentToolCall.from_part(
                                            agent_name, part
                                        )
                                    else:
                                        _loggy.debug(
                                            "Skipping unsupported part",
                                            result_type=type_name(part),
                                        )
                                        continue

                                    _loggy.debug(
                                        "Emitting part",
                                        uid=output.uid,
                                        part_kind=event.part.part_kind,
                                    )
                                    yield output

                                # Delta of part
                                elif isinstance(event, PartDeltaEvent):
                                    delta = event.delta
                                    if delta.part_delta_kind == "text":
                                        output = AgentText.from_delta(agent_name, delta)
                                    elif delta.part_delta_kind == "thinking":
                                        output = AgentThink.from_delta(
                                            agent_name, delta
                                        )
                                    elif delta.part_delta_kind == "tool_call":
                                        output = AgentToolCall.from_delta(
                                            agent_name, delta
                                        )
                                    else:
                                        _loggy.debug(
                                            "Skipping unsupported delta",
                                            result_type=type_name(delta),
                                        )
                                        continue

                                    _loggy.debug(
                                        "Emitting delta",
                                        uid=output.uid,
                                        delta_kind=event.delta.part_delta_kind,
                                    )
                                    yield output

                    # Tool call
                    elif Agent.is_call_tools_node(node):
                        async with node.stream(agent_run.ctx) as tool_stream:
                            async for event in tool_stream:
                                # Initial call
                                if isinstance(event, FunctionToolCallEvent):
                                    output = AgentToolCall.from_part(
                                        agent_name, event.part
                                    )
                                    _loggy.debug(
                                        "Emitting tool call",
                                        uid=output.uid,
                                        output_type=type(output).__name__,
                                    )
                                    yield output

                                # Final result
                                elif isinstance(event, FunctionToolResultEvent):
                                    if isinstance(event.result, ToolReturnPart):
                                        output = AgentToolReturn.from_part(
                                            agent_name, event.result
                                        )
                                        log_kwargs = {
                                            "uid": output.uid,
                                            "output_type": type_name(output),
                                        }
                                        if output.name is not None:
                                            log_kwargs["tool_name"] = output.name
                                        _loggy.debug(
                                            "Emitting tool return", **log_kwargs
                                        )
                                        yield output
                                    else:
                                        _loggy.debug(
                                            "Skipping unsupported tool result",
                                            result_type=type_name(event.result),
                                        )

                # Set result after graph iteration complete
                run_result = agent_run.result

        # Set messages after run is complete
        finally:
            if messages is not None and run_result is not None:
                new_messages = run_result.new_messages()
                _loggy.debug(
                    "Updating message history",
                    added_count=len(new_messages),
                )
                messages.extend(new_messages)

        if run_result is not None:
            final_output = cast(
                AgentFinal[OutputType],
                AgentFinal.from_run_result(agent_name, run_result),
            )
            _loggy.debug(
                "Emitting final result",
                uid=final_output.uid,
                output_type=type_name(final_output),
            )
            yield final_output

    return agent_stream


# --- Exports ---
__all__ = [
    "create_agent",
    "create_agent_stream",
]

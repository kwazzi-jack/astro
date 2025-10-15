"""
astro/agents/base.py

Core type definitions and base agent abstractions.

Author: Your Name
Date: 2025-07-27
License: MIT

Description:
    Provides protocols, type aliases, and abstract base classes for agent state and behavior.

Dependencies:
    - pydantic
"""

# --- Internal Imports ---
from collections.abc import AsyncIterator, Callable, Sequence
from types import NoneType
from typing import Any, Literal

from pydantic import BaseModel

# --- External Imports ---
from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponsePartDelta,
    ModelSettings,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPartDelta,
    ToolCallPartDelta,
    UserContent,
)
from pydantic_ai.models import Model

# --- Local Imports ---
from astro.llms import create_llm_model
from astro.llms.contexts import ChatContext, Context
from astro.llms.prompts import (
    create_assistant_message,
    get_prompt_template,
)
from astro.loggings.base import get_loggy
from astro.typings import NamedDict, type_name
from astro.utilities.display import pretty_print_model_message
from astro.utilities.timing import get_datetime_str

# --- Globals ---
loggy = get_loggy(__file__)


# ======================================================================
# Event Data Model
# ======================================================================
class ChatEvent(BaseModel):
    type: Literal[
        "start", "text", "thinking", "tool_call", "tool_result", "final", "end"
    ]
    content: str | Sequence[UserContent] | None = None
    name: str | None = None
    args: str | NamedDict | None = None
    result: Any | None = None


def _response_delta_to_chat_event(part: ModelResponsePartDelta) -> ChatEvent:
    match part.part_delta_kind:
        case "text":
            return ChatEvent(type="text", content=part.content_delta)
        case "thinking":
            return ChatEvent(type="thinking", content=part.content_delta)
        case "tool_call":
            return ChatEvent(
                type="tool_call", name=part.tool_name_delta, args=part.args_delta
            )
        case default:
            raise loggy.ValueError(f"Internal error: unknown {default}")


def _to_chat_event(event: AgentStreamEvent | AgentRunResultEvent[str]) -> ChatEvent:
    """Translate low-level AgentStreamEvent into ChatEvent."""
    if isinstance(event, PartStartEvent):
        return ChatEvent(type="start")

    if isinstance(event, PartDeltaEvent):
        return _response_delta_to_chat_event(event.delta)

    if isinstance(event, FunctionToolCallEvent):
        return ChatEvent(
            type="tool_call", name=event.part.tool_name, args=event.part.args
        )
    if isinstance(event, FunctionToolResultEvent):
        return ChatEvent(
            type="tool_result", content=str(event.content), result=event.result
        )

    if isinstance(event, FinalResultEvent):
        return ChatEvent(type="final", name=event.tool_name)

    if isinstance(event, AgentRunResultEvent):
        return ChatEvent(type="end", result=event.result)

    else:
        raise loggy.ValueError(f"Unknown event type: {type_name(event)}")


# ======================================================================
# Agent Factory
# ======================================================================
def create_agent(
    identifier: str,
    instructions: str | Sequence[str] | None = None,
    model_settings: ModelSettings | None = None,
    context_type: type = NoneType,
    agent_name: str = "agent",
) -> Agent[Context, str]:
    try:
        model = create_llm_model(identifier)
    except Exception as error:
        raise loggy.CreationError(object_type=Model, caused_by=error)
    return Agent(
        name=agent_name,
        model=model,
        instructions=instructions,
        deps_type=context_type,
        model_settings=model_settings,
    )


# ======================================================================
# Chat Constructor
# ======================================================================
def create_astro_chat(
    identifier: str = "ollama:llama3.1:latest",
) -> tuple[
    Callable[[str], AsyncIterator[AgentStreamEvent | AgentRunResultEvent[str]]],
    list[ModelMessage],
]:
    astro_agent = create_agent(
        identifier,
        model_settings=ModelSettings(temperature=0.7),
        context_type=ChatContext,
        agent_name="astro-chat",
    )
    context = ChatContext()
    system_template = get_prompt_template("chat-system")
    welcome_template = get_prompt_template("chat-welcome")

    messages: list[ModelMessage] = [create_assistant_message(welcome_template(context))]

    @astro_agent.instructions
    async def get_system_instructions(ctx: RunContext[ChatContext]) -> str:
        return system_template(ctx.deps)

    @astro_agent.tool(docstring_format="google")
    async def get_system_datetime(ctx: RunContext[ChatContext]) -> str:
        """Get the current system datetime formatted as a string.

        Returns a formatted datetime string from the chat context's datetime attribute.

        Args:
            `ctx` (`RunContext[ChatContext]`): Runtime context containing chat state and dependencies.

        Returns:
            `str`: Formatted datetime string representing the current system time.
        """
        return get_datetime_str(ctx.deps.datetime)

    async def chat_stream(
        prompt: str,
    ) -> AsyncIterator[AgentStreamEvent | AgentRunResultEvent[str]]:
        nonlocal messages
        async for event in astro_agent.run_stream_events(
            prompt, message_history=messages, deps=context
        ):
            if event.event_kind == "agent_run_result":
                messages.extend(event.result.new_messages())
            yield event

    return chat_stream, messages


# ======================================================================
# Main Interactive Loop
# ======================================================================
async def main():
    from rich.console import Group
    from rich.live import Live
    from rich.markdown import Markdown
    from rich.panel import Panel

    from astro.utilities.display import user_md_input

    chat, messages = create_astro_chat()

    # Styling map
    STYLE_MAP = {
        "text": "purple",
        "thinking": "yellow",
        "tool_call": "blue",
        "tool_result": "cyan",
        "final": "green",
        "end": "red",
    }

    def create_panel(title: str = "", content: str = "", etype: str = "") -> Panel:
        return Panel(
            Markdown(content),
            title=title,
            title_align="left",
            style=STYLE_MAP.get(etype, "dim"),
            expand=True,
        )

    while True:
        user_prompt = user_md_input()
        if user_prompt.lower() in ("q", "quit"):
            break
        if user_prompt.lower() == "/messages":
            for m in messages:
                pretty_print_model_message(m)
            continue

        panels: list[Panel] = []
        current_text = ""

        def render_screen() -> Group:
            return Group(*panels)

        with Live(render_screen(), refresh_per_second=8, transient=False) as live:
            current_content = ""
            counter = 0
            was_tool_call = False
            offset = 0
            async for event in chat(user_prompt):
                if isinstance(event, PartStartEvent):
                    if event.part.part_kind == "text" and len(event.part.content) > 0:
                        current_content = event.part.content
                        panels.insert(
                            event.index,
                            create_panel("Astro", current_content, "text"),
                        )

                    elif event.part.part_kind == "thinking":
                        current_content = event.part.content
                        panels.insert(
                            event.index,
                            create_panel("Thinking", current_content, "thinking"),
                        )

                    elif event.part.part_kind == "tool-call":
                        current_content = str(event.part.args)
                        panels.insert(
                            event.index,
                            create_panel(
                                event.part.tool_name, current_content, "tool_call"
                            ),
                        )

                elif isinstance(event, PartDeltaEvent):
                    if (
                        event.delta.part_delta_kind == "text"
                        and len(event.delta.content_delta) > 0
                    ):
                        current_content += event.delta.content_delta
                        panel = create_panel("Astro", current_content, "text")
                        panels[event.index] = panel
                    elif event.delta.part_delta_kind == "thinking":
                        current_content += str(event.delta.content_delta)
                        panel = create_panel("Thinking", current_content, "thinking")
                        panels[event.index] = panel
                    elif event.delta.part_delta_kind == "tool_call":
                        current_content += "Calling"
                        if (
                            event.delta.args_delta is not None
                            and len(event.delta.args_delta) > 0
                        ):
                            current_content += (
                                f" with arguments `{event.delta.args_delta}`"
                            )
                        panel = create_panel("Tool Call", current_content, "tool_call")
                        panels[event.index] = panel

                live.update(render_screen())
                counter += 1
                # print(f"Event {counter}: {event}")
                # print("=" * 40)
                # print(
                #     "\n".join(
                #         [
                #             f" -> {counter}.{i} - {panel.title}"
                #             for i, panel in enumerate(panels)
                #         ]
                #     )
                # )
                # print("=" * 40)


# ======================================================================
# Entrypoint
# ======================================================================
if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Astro Chat...")

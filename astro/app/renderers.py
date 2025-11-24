"""Rich renderable factories for presenting agent stream outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from rich.console import RenderableType
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from astro.theme import BORDER_COLOR, TEXT_DIM, TEXT_PRIMARY, TEXT_SECONDARY
from astro.typings.outputs import AgentToolCall, AgentToolReturn


@dataclass
class ToolInteractionRenderData:
    """Container describing an agent tool interaction lifecycle.

    Attributes:
        call (AgentToolCall | None): Latest tool call details captured for the
            interaction.
        result (AgentToolReturn | None): Matched tool return payload when
            available.
    """

    call: AgentToolCall | None = None
    result: AgentToolReturn | None = None


def render_agent_text(content: str, *, style: str = TEXT_PRIMARY) -> Text:
    """Create a Rich Text renderable for agent text chunks.

    Args:
        content (str): Text fragment emitted by the agent. May contain Rich
            markup.
        style (str): Rich style token applied when rendering the text.

    Returns:
        Text: Renderable configured to display the agent text content.
    """

    rendered_text = Text.from_markup(content, style=style)
    rendered_text.overflow = "fold"
    return rendered_text


def render_agent_think(
    content: str,
    *,
    signature: str | None = None,
    body_style: str = TEXT_SECONDARY,
    border_style: str = BORDER_COLOR,
) -> Panel:
    """Create a Rich Panel renderable for agent thinking traces.

    Args:
        content (str): Thinking content emitted by the agent.
        signature (str | None): Optional signature describing the thought step.
        body_style (str): Rich style applied to the panel body text.
        border_style (str): Rich style applied to the panel border.

    Returns:
        Panel: Renderable encapsulating the thinking content within a panel.
    """

    body = Text.from_markup(content, style=body_style)
    body.overflow = "fold"
    return Panel(body, title=signature, title_align="left", border_style=border_style)


def render_tool_interaction(
    interaction: ToolInteractionRenderData,
    *,
    value_color: str = TEXT_PRIMARY,
    pending_color: str = TEXT_DIM,
) -> RenderableType:
    """Render the state of a tool interaction using spinner and panels.

    Args:
        interaction (ToolInteractionRenderData): Aggregate tool interaction data.
        value_color (str): Style applied to the rendered text payload.
        pending_color (str): Style applied while awaiting a result.

    Returns:
        RenderableType: Spinner while waiting for results, otherwise a panel
            containing the tool output.
    """

    call = interaction.call
    result = interaction.result
    signature = _format_tool_signature(call)

    if result is None:
        spinner_style = pending_color if pending_color else value_color
        spinner_text = Text(signature, style=value_color)
        return Spinner("dots", text=spinner_text, style=spinner_style)

    result_payload = _create_value_text(result.content, value_color, with_repr=True)
    panel_title = signature
    if result.name and (call is None or result.name != call.name):
        panel_title = f"{signature} → {result.name}"

    return Panel(
        result_payload,
        title=panel_title,
        title_align="left",
        border_style=BORDER_COLOR,
        padding=(0, 1),
    )


def _create_value_text(value: Any, style: str, with_repr: bool = False) -> Text:
    """Convert an arbitrary value into a styled Rich Text instance.

    Args:
        value (Any): Object that should be rendered inside the table cell.
        style (str): Rich style expression applied to the resulting text.
        with_repr (bool): Whether to use the repr() form for strings.

    Returns:
        Text: Rich text instance containing the formatted payload.
    """

    as_string = _stringify_payload(value, with_repr=with_repr)
    text = Text(as_string, style=style)
    text.overflow = "fold"
    return text


def _format_tool_signature(call: AgentToolCall | None) -> str:
    """Return a human-friendly representation of a tool invocation.

    Args:
        call (AgentToolCall | None): Tool call payload captured from the agent.

    Returns:
        str: Stringified signature suitable for inline display.
    """

    if call is None:
        return "tool()"

    name = call.name or "tool"
    arguments = call.arguments
    argument_values = []
    if arguments is None:
        argument_values = []
    elif isinstance(arguments, dict):
        argument_values = list(arguments.values())
    elif isinstance(arguments, (list, tuple)):
        argument_values = list(arguments)
    else:
        argument_values = [arguments]

    rendered_args = ", ".join(
        _stringify_payload(value, with_repr=True) for value in argument_values
    )
    return f"{name}({rendered_args})" if rendered_args else f"{name}()"


def _stringify_payload(payload: Any, with_repr: bool = False) -> str:
    """Return a human-readable string representation for a payload.

    Args:
        payload (Any): Value originating from a tool call argument or result.
        with_repr (bool): Whether to use the repr() form for strings.

    Returns:
        str: String representation suitable for console rendering.
    """

    if payload is None:
        return "--"
    if isinstance(payload, str):
        if with_repr:
            return f'"""{payload}"""' if r"\n" in payload else f'"{payload}"'
        else:
            return payload
    try:
        return json.dumps(payload, indent=2, ensure_ascii=False)
    except TypeError:
        if with_repr:
            return repr(payload)
        else:
            return str(payload)

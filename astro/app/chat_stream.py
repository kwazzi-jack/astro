"""Reusable chat streaming renderer utilities for Astro."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import Enum, auto

from rich.console import Console, RenderableType
from rich.live import Live

from astro.app.renderers import (
    ToolInteractionRenderData,
    render_agent_text,
    render_agent_think,
    render_tool_interaction,
)
from astro.theme import RuleStyle, build_response_start_rule_style
from astro.typings.callables import StreamFn
from astro.typings.outputs import (
    AgentFinal,
    AgentOutput,
    AgentText,
    AgentThink,
    AgentToolCall,
    AgentToolReturn,
)


class BlockKind(Enum):
    """Enumerate supported stream block categories."""

    TEXT = auto()
    THINK = auto()
    TOOL = auto()


class ChatBlock:
    """Incrementally construct a renderable for a single streamed block."""

    kind: BlockKind

    def accepts(self, output: AgentOutput) -> bool:  # pragma: no cover - interface
        """Return whether the block can consume the provided output.

        Args:
            output (AgentOutput): Streamed output to evaluate.

        Returns:
            bool: True when the block accepts the output type.
        """

        raise NotImplementedError

    def update(self, output: AgentOutput) -> bool:  # pragma: no cover - interface
        """Apply a streamed event to the block state.

        Args:
            output (AgentOutput): Streamed output to consume.

        Returns:
            bool: True when the block state changed.
        """

        raise NotImplementedError

    def render(self) -> RenderableType:  # pragma: no cover - interface
        """Return the Rich renderable representing the block.

        Returns:
            RenderableType: Rich renderable describing the block contents.
        """

        raise NotImplementedError

    def has_payload(self) -> bool:  # pragma: no cover - interface
        """Return True when the block accumulated visible content.

        Returns:
            bool: True when the block contains content to display.
        """

        raise NotImplementedError

    def should_finalize(self) -> bool:
        """Return True when the block should be closed after the update.

        Returns:
            bool: True when the renderer should close the block.
        """

        return False


class TextBlock(ChatBlock):
    """Accumulate agent text output for a block with a styled prefix."""

    def __init__(self, prefix_markup: str) -> None:
        """Initialise the text block.

        Args:
            prefix_markup (str): Rich markup prefix applied to the text block.
        """

        self.kind = BlockKind.TEXT
        self._prefix_markup = prefix_markup
        self._content = ""

    def accepts(self, output: AgentOutput) -> bool:
        """Return True when the output is a text chunk."""

        return isinstance(output, AgentText)

    def update(self, output: AgentOutput) -> bool:
        """Append streamed text to the block.

        Args:
            output (AgentOutput): Streamed agent output containing text.

        Returns:
            bool: True when the block changed.
        """

        assert isinstance(output, AgentText)
        if len(output.text) == 0:
            return False
        self._content += output.text
        return True

    def render(self) -> RenderableType:
        """Return the renderable for the current text state.

        Returns:
            RenderableType: Rich renderable containing text content.
        """

        content = f"{self._prefix_markup}{self._content}"
        return render_agent_text(content)

    def has_payload(self) -> bool:
        """Return True when content has been captured.

        Returns:
            bool: True when accumulated text is non-empty.
        """

        return len(self._content) > 0


class ThinkBlock(ChatBlock):
    """Accumulate agent thinking traces for display in a panel."""

    def __init__(self, heading_markup: str) -> None:
        """Initialise the think block.

        Args:
            heading_markup (str): Styled heading describing the think block.
        """

        self.kind = BlockKind.THINK
        self._heading_markup = heading_markup
        self._content = ""
        self._signature: str | None = None

    def accepts(self, output: AgentOutput) -> bool:
        """Return True when the output is a think chunk."""

        return isinstance(output, AgentThink)

    def update(self, output: AgentOutput) -> bool:
        """Append streamed thinking content to the block.

        Args:
            output (AgentOutput): Streamed agent output representing thinking.

        Returns:
            bool: True when the block changed.
        """

        assert isinstance(output, AgentThink)
        new_text = output.text or ""
        if len(new_text) == 0:
            return False
        self._content += new_text
        if output.signature:
            self._signature = output.signature
        return True

    def render(self) -> RenderableType:
        """Return the renderable panel describing the thinking block.

        Returns:
            RenderableType: Rich renderable representing the thinking panel.
        """

        signature = self._heading_markup
        if self._signature is not None:
            signature = f"{self._heading_markup} · {self._signature}"
        return render_agent_think(self._content, signature=signature)

    def has_payload(self) -> bool:
        """Return True when thinking content has been captured.

        Returns:
            bool: True when accumulated thinking text is non-empty.
        """

        return len(self._content) > 0


class ToolBlock(ChatBlock):
    """Accumulate tool call and result information for a block."""

    def __init__(self) -> None:
        """Initialise the tool block."""

        self.kind = BlockKind.TOOL
        self._state = ToolInteractionRenderData()
        self._call_id: str | None = None

    def accepts(self, output: AgentOutput) -> bool:
        """Return True when the output matches the tracked tool call."""

        if not isinstance(output, (AgentToolCall, AgentToolReturn)):
            return False
        candidate = output.call_id or getattr(output, "uid", None)
        if candidate is None:
            return True
        return self._call_id in (None, candidate)

    def update(self, output: AgentOutput) -> bool:
        """Update tool interaction state from streamed output.

        Args:
            output (AgentOutput): Tool call or result emitted by the agent.

        Returns:
            bool: True when the block changed.
        """

        assert isinstance(output, (AgentToolCall, AgentToolReturn))
        identifier = output.call_id or getattr(output, "uid", None)
        if self._call_id is None and identifier is not None:
            self._call_id = identifier
        if isinstance(output, AgentToolCall):
            self._state.call = output
        else:
            self._state.result = output
        return True

    def render(self) -> RenderableType:
        """Return the renderable depicting the tool interaction.

        Returns:
            RenderableType: Rich renderable describing the tool interaction.
        """

        return render_tool_interaction(self._state)

    def has_payload(self) -> bool:
        """Return True when a call or result has been registered.

        Returns:
            bool: True when either a call or result is present.
        """

        return self._state.call is not None or self._state.result is not None

    def should_finalize(self) -> bool:
        """Return True after the tool result has been received.

        Returns:
            bool: True once the tool result has been captured.
        """

        return self._state.result is not None


class ChatStreamRenderer:
    """Render a chat agent stream as sequential Rich blocks."""

    def __init__(
        self,
        console: Console,
        prefix_factory: Callable[[], str],
        think_prefix_factory: Callable[[], str] | None = None,
        refresh_per_second: int = 12,
    ) -> None:
        """Initialise the chat stream renderer.

        Args:
            console (Console): Rich console used for rendering output.
            prefix_factory (Callable[[], str]): Callable returning the styled
                agent prefix string for text blocks.
            think_prefix_factory (Callable[[], str] | None): Optional callable
                returning the styled prefix for think blocks. Defaults to None,
                in which case the text prefix factory is reused.
            refresh_per_second (int): Refresh rate for the live view. Defaults
                to 12.
        """

        self._console = console
        self._prefix_factory = prefix_factory
        self._think_prefix_factory = think_prefix_factory or prefix_factory
        self._refresh_per_second = refresh_per_second

    async def render(self, prompt: str, stream_fn: StreamFn) -> None:
        """Stream the chat agent response and render it live.

        Args:
            prompt (str): Prompt routed to the agent for completion.
            stream_fn (StreamFn): Async callable yielding agent output events.

        Raises:
            ValueError: Raised when stream_fn is None.

        Returns:
            None: This method does not return a value.
        """

        if stream_fn is None:
            raise ValueError("Chat stream function is not initialised")

        stream = stream_fn(prompt)
        current_block: ChatBlock | None = None
        live: Live | None = None
        rule_printed = False
        text_payload_rendered = False

        def finalize_block() -> None:
            """Stop the live rendering for the current block.

            Returns:
                None: This helper does not return a value.
            """
            nonlocal current_block, live, rule_printed
            had_payload = (
                current_block.has_payload() if current_block is not None else False
            )
            if live is not None:
                live.__exit__(None, None, None)
                live = None
            if had_payload and rule_printed:
                self._console.print()
            current_block = None
            rule_printed = False

        def start_block(kind: BlockKind) -> None:
            """Initialise a new block and start the live renderer.

            Args:
                kind (BlockKind): Block category to start streaming.

            Returns:
                None: This helper does not return a value.
            """

            nonlocal current_block, live, rule_printed
            finalize_block()
            current_block = self._create_block(kind)
            rule_printed = False
            live_renderable = current_block.render()
            live = Live(
                live_renderable,
                console=self._console,
                refresh_per_second=self._refresh_per_second,
            )
            live.__enter__()
            live.refresh()

        try:
            async for output in stream:
                if isinstance(output, AgentText) and len(output.text) == 0:
                    continue
                kind = self._classify_output(output)
                if kind is None:
                    continue
                if current_block is None:
                    start_block(kind)
                if isinstance(output, AgentFinal):
                    final_output = output.output
                    if (
                        isinstance(final_output, str)
                        and len(final_output) > 0
                        and not text_payload_rendered
                    ):
                        if not isinstance(current_block, TextBlock):
                            if current_block is not None:
                                finalize_block()
                            start_block(BlockKind.TEXT)
                        if isinstance(current_block, TextBlock):
                            text_event = AgentText(
                                agent_name=output.agent_name,
                                text=final_output,
                            )
                            if current_block.update(text_event) and live is not None:
                                live.update(current_block.render())
                                if current_block.has_payload():
                                    text_payload_rendered = True
                    finalize_block()
                    continue

                if kind == BlockKind.TOOL and isinstance(current_block, ToolBlock):
                    if not current_block.accepts(output):
                        finalize_block()
                        current_block = None
                if current_block is None:
                    if kind == BlockKind.TEXT:
                        start_block(BlockKind.TEXT)
                    elif kind == BlockKind.THINK:
                        start_block(BlockKind.THINK)
                    elif kind == BlockKind.TOOL:
                        start_block(BlockKind.TOOL)
                    else:
                        continue
                elif current_block.kind != kind:
                    finalize_block()
                    if kind == BlockKind.TEXT:
                        start_block(BlockKind.TEXT)
                    elif kind == BlockKind.THINK:
                        start_block(BlockKind.THINK)
                    elif kind == BlockKind.TOOL:
                        start_block(BlockKind.TOOL)
                    else:
                        continue

                assert current_block is not None
                if not current_block.accepts(output):
                    finalize_block()
                    start_block(kind)
                updated = current_block.update(output)
                if not updated:
                    continue
                if not rule_printed and current_block.has_payload():
                    self._console.print()
                    self._render_rule(build_response_start_rule_style(datetime.now()))
                    self._console.print()
                    rule_printed = True
                if live is not None:
                    live.update(current_block.render())
                if (
                    kind == BlockKind.TEXT
                    and isinstance(current_block, TextBlock)
                    and current_block.has_payload()
                ):
                    text_payload_rendered = True
                if current_block.should_finalize():
                    finalize_block()
        finally:
            finalize_block()

    def _create_block(self, kind: BlockKind) -> ChatBlock:
        """Return a block implementation for the requested kind.

        Args:
            kind (BlockKind): Block category to construct.

        Returns:
            ChatBlock: Concrete block implementation.

        Raises:
            ValueError: Raised when the block kind is unsupported.
        """

        if kind == BlockKind.TEXT:
            return TextBlock(self._prefix_factory())
        if kind == BlockKind.THINK:
            return ThinkBlock(self._think_prefix_factory())
        if kind == BlockKind.TOOL:
            return ToolBlock()
        raise ValueError(f"Unsupported block kind: {kind}")

    def _classify_output(self, output: AgentOutput) -> BlockKind | None:
        """Return the block kind represented by the streamed output.

        Args:
            output (AgentOutput): Streamed output to classify.

        Returns:
            BlockKind | None: Block category when recognised, otherwise None.
        """

        if isinstance(output, AgentText):
            return BlockKind.TEXT
        if isinstance(output, AgentThink):
            return BlockKind.THINK
        if isinstance(output, (AgentToolCall, AgentToolReturn)):
            return BlockKind.TOOL
        if isinstance(output, AgentFinal):
            return None
        return None

    def _render_rule(self, rule_style: RuleStyle) -> None:
        """Render a Rich rule using the provided style.

        Args:
            rule_style (RuleStyle): Descriptor describing the rule styling.

        Returns:
            None: This helper does not return a value.
        """

        style_value = rule_style.style
        if style_value is None:
            self._console.rule(
                rule_style.text,
                characters=rule_style.character,
                align=rule_style.align,
            )
        else:
            self._console.rule(
                rule_style.text,
                characters=rule_style.character,
                align=rule_style.align,
                style=str(style_value),
            )
        if rule_style.newline:
            self._console.print()


__all__ = ["ChatStreamRenderer"]

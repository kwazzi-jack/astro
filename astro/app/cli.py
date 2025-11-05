"""CLI entrypoint for the Astro interactive shell."""

# --- Internal Imports ---
import inspect
from dataclasses import dataclass
from typing import Any

# --- External Imports ---
import docstring_parser
from prompt_toolkit import PromptSession
from prompt_toolkit.lexers import PygmentsLexer
from rich.console import Console, RenderableType
from rich.console import Group as RichGroup
from rich.live import Live
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

# --- Local Imports ---
from astro.agents.chat import create_astro_chat
from astro.app.state import _AppState
from astro.logger import get_loggy
from astro.paths import APPSTATE_PATH
from astro.theme import (
    PRIMARY_COLOR,
    PRIMARY_LIGHT,
    SECONDARY_COLOR,
    SECONDARY_LIGHT,
    TEXT_DIM,
    TEXT_SECONDARY,
    AstroStyle,
    PromptConfiguration,
    RuleStyle,
    build_error_banner,
    build_exit_rule_style,
    build_response_end_rule_style,
    build_response_start_rule_style,
    create_astro_prompt_style,
    create_prompt_configuration,
    create_ptk_prompt_prefix_factory,
    create_rich_prompt_prefix,
    create_system_prompt_style,
    escape_rich_markup,
    get_welcome_header,
    html_obj_to_rich_format,
    make_rich_theme,
    prompt_model_selection,
)
from astro.typings import AnyFactory, AsyncChatFunction, MessageList
from astro.utilities.timing import create_timer, get_datetime_now, seconds_to_strtime

# --- GLOBALS ---
_loggy = get_loggy(__file__)


def _get_docstring_short_description(obj: Any) -> str:
    """Return the short description extracted from an object's docstring.

    Args:
        obj (Any): Object to inspect.

    Returns:
        str: Short description parsed from the object's docstring.

    Raises:
        ValueError: Raised when the object lacks a docstring or short
            description.
    """
    if not hasattr(obj, "__doc__"):
        raise _loggy.ValueError(f"Cannot get docstring from {obj!r} with no docstring")
    result = docstring_parser.parse_from_object(obj).short_description
    if result is None:
        raise _loggy.ValueError(f"Could not find short-description for {obj!r}")
    return result


def _get_command_help_info(obj: Any) -> tuple[str, str]:
    """Extract short and long descriptions from a command handler's docstring.

    Args:
        obj (Any): Command handler object to inspect.

    Returns:
        tuple[str, str]: Short description and long description (if available).

    Raises:
        ValueError: Raised when the object lacks a docstring.
    """
    if not hasattr(obj, "__doc__"):
        raise _loggy.ValueError(f"Cannot get docstring from {obj!r} with no docstring")

    parsed = docstring_parser.parse_from_object(obj)
    short_desc = parsed.short_description or "No description available"
    long_desc = parsed.long_description or ""

    return short_desc, long_desc


@dataclass(frozen=True)
class CommandSpec:
    """Describe a CLI command, its handler, and short summary.

    Attributes:
        name (str): Command token (for example "/help").
        handler (CommandHandler): Handler invoked when the command is run.
        summary (str): Short description displayed in help and completions.
    """

    name: str
    handler: AnyFactory
    summary: str

    @classmethod
    def from_handler(cls, name: str, handler: AnyFactory) -> "CommandSpec":
        """Create a command specification from a handler.

        Args:
            name (str): Command token for registration.
            handler (CommandHandler): Handler executed when the command is
                triggered.

        Returns:
            CommandSpec: Registered command specification.
        """

        summary = _get_docstring_short_description(handler)
        return cls(name=name, handler=handler, summary=summary)


# -- CLI ---
class AstroCLI:
    """Interactive shell for Astro commands."""

    def __init__(self, overwrite_state: bool = False):
        """Initialise the Astro CLI shell.

        Args:
            overwrite_state (bool): Optional flag indicating whether to
                overwrite the existing persisted application state when missing.
                Defaults to False.
        """

        self._state = self._load_appstate(overwrite_state)
        self._console = Console(theme=make_rich_theme())

        # Timer functions
        start_timer, stop_timer = create_timer()
        self._start_timer_fn = start_timer
        self._stop_timer_fn = stop_timer

        # Maintain a registry of slash commands and their handlers for routing.
        self._command_map = self._register_commands()
        # Hashtag metadata powers context completion and validation.
        self._hashtags = self._build_hashtag_registry()

        # Assemble prompt configuration via theming helpers to centralise UI.
        self._prompt_config = create_prompt_configuration(
            username_provider=lambda: self._state.username,
            model_identifier_provider=lambda: self._state.current_model.to_identifier(),
            datetime_provider=lambda: get_datetime_now(to_local=True),
            command_metadata={
                name: spec.summary for name, spec in self._command_map.items()
            },
            hashtag_metadata=self._hashtags,
        )
        self._session = self._create_prompt_session(self._prompt_config)
        self._chat_stream: AsyncChatFunction | None = None
        self._chat_history: MessageList | None = None
        self._initialise_chat_agent()
        self._previous_response_duration: float | None = None
        self._astro_prompt_style: AstroStyle = create_astro_prompt_style()
        self._system_prompt_style: AstroStyle = create_system_prompt_style()

    def _load_appstate(self, overwrite: bool = False) -> _AppState:
        """Load the persisted application state or create a new one.

        Args:
            overwrite (bool): Optional flag indicating whether to overwrite
                existing state if the file is missing. Defaults to False.

        Returns:
            _AppState: Loaded or newly created application state.

        Raises:
            ValueError: Raised when state loading or creation fails due to
                validation issues or missing files.
        """

        try:
            return _AppState.load()
        except ValueError as error:
            raise _loggy.LoadError(
                path_or_uid=APPSTATE_PATH,
                obj_or_key=_AppState,
                load_from="json",
                caused_by=error,
            )
        except FileNotFoundError:
            try:
                return _AppState.touch(overwrite)
            except Exception as error:  # pragma: no cover - defensive guard
                raise _loggy.CreationError(
                    object_type=_AppState,
                    reason="Tried to create new file",
                    caused_by=error,
                )
        except Exception as error:  # pragma: no cover - defensive guard
            raise _loggy.LoadError(
                path_or_uid=APPSTATE_PATH,
                obj_or_key=_AppState,
                load_from="json",
                caused_by=error,
            )

    def _register_commands(self) -> dict[str, CommandSpec]:
        """Register the built-in slash command handlers.

        Returns:
            dict[str, CommandSpec]: Mapping from command token to specification.
        """

        specs = [
            CommandSpec.from_handler("/init", self._cmd_init),
            CommandSpec.from_handler("/help", self._cmd_help),
            CommandSpec.from_handler("/quit", self._cmd_exit),
            CommandSpec.from_handler("/welcome", self._cmd_welcome),
            CommandSpec.from_handler("/model", self._cmd_model),
        ]
        return {spec.name: spec for spec in specs}

    def _build_hashtag_registry(self) -> dict[str, str]:
        """Return the hashtag metadata used for completions.

        Returns:
            dict[str, str]: Mapping from hashtag token to description.
        """

        return {
            "#context": "Reference conversation context",
            "#history": "Reference chat history",
            "#system": "Reference system information",
        }

    def _create_prompt_session(self, config: PromptConfiguration) -> PromptSession:
        """Create the prompt session configured with theming helpers.

        Args:
            config (PromptConfiguration): Prompt configuration generated by the
                theme module.

        Returns:
            PromptSession: Configured prompt session instance.
        """

        return PromptSession(
            message=config.prompt_prefix,
            placeholder=config.placeholder,
            bottom_toolbar=config.bottom_toolbar,
            color_depth=config.color_depth,
            completer=config.completer,
            complete_style=config.complete_style,
            complete_in_thread=config.complete_in_thread,
            complete_while_typing=config.complete_while_typing,
            validator=config.validator,
            validate_while_typing=config.validate_while_typing,
            lexer=PygmentsLexer(config.lexer_cls),
            style=config.style,
            refresh_interval=config.refresh_interval,
            wrap_lines=config.wrap_lines,
            cursor=config.cursor_shape,
            erase_when_done=True,
        )

    def _initialise_chat_agent(self) -> None:
        """Initialise the chat agent for the currently selected model.

        Returns:
            None: This method does not return a value.

        Raises:
            CreationError: Raised when the chat agent fails to initialise for
                the selected model identifier.
        """

        identifier = self._state.current_model.to_identifier()
        try:
            stream_fn, message_history = create_astro_chat(identifier)
        except Exception as error:  # pragma: no cover - defensive guard
            _loggy.exception(
                "Failed to initialise chat agent",
                identifier=identifier,
            )
            self._chat_stream = None
            self._chat_history = None
            raise _loggy.CreationError(
                object_type="chat agent",
                reason=f"Initialisation failed for {identifier}",
                caused_by=error,
            )

        self._chat_stream = stream_fn
        self._chat_history = message_history

    def _render_rule(self, rule_style: RuleStyle) -> None:
        """Render a Rich rule using the provided style.

        Args:
            rule_style (RuleStyle): Descriptor describing the rule styling.
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

    def _print_timestamp(
        self,
        elapsed_seconds: float,
        prefix_text: str = "Δt ~ ",
        suffix_text: str = "",
        separator: str = "",
        padding: int | tuple[int, int] = 0,
    ) -> None:
        """Print a timestamp with the elapsed time.

        Args:
            elapsed_seconds (float): Elapsed time in seconds.
            prefix_text (str): Custom text to display before the timestamp.
            suffix_text (str): Custom text to display after the timestamp.
            padding (int | tuple[int, int]): Padding around the timestamp.
        """
        if isinstance(padding, int):
            left_pad = right_pad = " " * padding
        else:
            left_pad, right_pad = " " * padding[0], " " * padding[1]

        formatted_time = seconds_to_strtime(elapsed_seconds)
        self._console.print(
            f"{left_pad}{prefix_text}{separator}{formatted_time}{separator}{suffix_text}{right_pad}",
            style=f"italic {TEXT_DIM}",
            highlight=False,
        )

    def _start_timer(self) -> None:
        """Start the response timer."""

        self._start_timer_fn()

    def _stop_timer(self) -> float:
        """Stop the response timer and return elapsed seconds.

        Returns:
            float: Elapsed time in seconds.
        """

        return self._stop_timer_fn()

    def _print_newline(self) -> None:
        """Print an empty line to the console."""
        self._console.print()

    async def _get_input(self, as_block: bool = True) -> str:
        """Prompt the user for input and optionally surround with blank lines.

        Args:
            as_block (bool): Whether to print blank lines before and after the
                prompt. Defaults to True.

        Returns:
            str: Raw user input captured from the prompt.
        """
        if as_block:
            self._print_newline()
        result = await self._session.prompt_async()
        if as_block:
            self._print_newline()
        return result

    def _print(self, obj: RenderableType, as_block: bool = True) -> None:
        """Print an object with optional spacing before and after.

        Args:
            obj (RenderableType): Rich renderable to print.
            as_block (bool): Whether to surround the output with blank lines.
                Defaults to True.
        """
        if as_block:
            self._print_newline()
        self._console.print(obj)
        if as_block:
            self._print_newline()

    def _print_md(self, obj: str, as_block: bool = True) -> None:
        """Print markdown content with Rich formatting.

        Args:
            obj (str): Markdown text to render and print.
            as_block (bool): Whether to surround the output with blank lines.
                Defaults to True.
        """
        self._print(Markdown(obj), as_block=as_block)

    def _print_error(self, message: RenderableType) -> None:
        """Print an error banner with consistent styling.

        Args:
            message (RenderableType): Error message or renderable to display.
        """
        self._print(build_error_banner(str(message)))

    async def _handle_user_prompt(self, prompt: str) -> str:
        """Send a free-form prompt to the chat agent and stream its response.

        Args:
            prompt (str): User prompt to route to the agent.

        Returns:
            None: This method does not return a value.
        """
        prefix = html_obj_to_rich_format(self._prompt_config.prompt_prefix())
        user_text = Text.from_markup(
            f"{prefix}[{TEXT_SECONDARY}]{prompt}[/{TEXT_SECONDARY}]"
        )
        self._print(user_text, as_block=False)
        return prompt

    async def _stream_chat_response(self, prompt: str) -> None:
        """Stream the chat agent response for the provided prompt.

        Args:
            prompt (str): User prompt routed to the agent for completion.

        Returns:
            None: This method does not return a value.
        """

        if self._chat_stream is None:
            self._print_error(
                "Chat agent is not initialised. Try selecting a different model."
            )
            return

        response_buffer = create_rich_prompt_prefix(
            name="astro",
            name_style=self._astro_prompt_style,
            symbol_style=self._system_prompt_style,
            append_spaces=1,
        )
        has_streamed_content = False

        def _render_response() -> Markdown:
            return Markdown(response_buffer)

        with Live(
            _render_response(),
            console=self._console,
            refresh_per_second=12,
            get_renderable=_render_response,
        ) as live:
            try:
                async for event in self._chat_stream(prompt):
                    if event.event_kind == "part_start":
                        part = getattr(event, "part", None)
                        if getattr(part, "part_kind", None) == "text":
                            content = getattr(part, "content", "")
                            if content:
                                response_buffer += content
                                has_streamed_content = True
                                live.update(_render_response())
                    elif event.event_kind == "part_delta":
                        delta = getattr(event, "delta", None)
                        if getattr(delta, "part_delta_kind", None) == "text":
                            delta_content = getattr(delta, "content_delta", "")
                            if delta_content:
                                response_buffer += delta_content
                                has_streamed_content = True
                                live.update(_render_response())
                    elif event.event_kind == "agent_run_result":
                        output = event.result.output
                        if isinstance(output, str) and not has_streamed_content:
                            response_buffer += output
                            has_streamed_content = True
                            live.update(_render_response())
            except Exception as error:  # pragma: no cover - defensive guard
                live.stop()
                _loggy.exception(
                    "Chat streaming failed",
                    prompt=prompt,
                )
                self._print_error(f"Error while streaming response: {error}")
                return

        self._print_newline()

    def _build_help_text(self) -> RenderableType:
        """Build dynamic help text from registered command handlers.

        Returns:
            RenderableType: Formatted help panel with command information.
        """

        # Calculate the maximum width needed for first column
        max_cmd_width = max(len(cmd) for cmd in self._command_map.keys())
        max_tag_width = max(len(tag) for tag in self._hashtags.keys())
        first_col_width = (
            max(max_cmd_width, max_tag_width) + 4
        )  # +4 for "  " padding and markup

        # Create commands table
        cmd_table = Table(
            show_header=True,
            show_edge=False,
            pad_edge=False,
            box=None,
            padding=(0, 2),
            expand=False,  # Changed from True to False
        )
        cmd_table.add_column(
            "Commands",
            header_style=f"bold {TEXT_SECONDARY}",
            style=f"bold {SECONDARY_COLOR}",
            width=first_col_width,
            no_wrap=False,
        )
        cmd_table.add_column(
            "Description", header_style=f"bold {TEXT_SECONDARY}", style=SECONDARY_LIGHT
        )

        # Add each command to the table
        for cmd_name in sorted(self._command_map.keys()):
            spec = self._command_map[cmd_name]
            cmd_table.add_row(f"  {cmd_name}", f"  {spec.summary}")

        # Create hashtags table if there are any
        hashtag_table = Table(
            show_header=True,
            show_edge=False,
            pad_edge=False,
            box=None,
            padding=(0, 2),
            expand=False,  # Changed from True to False
        )
        hashtag_table.add_column(
            "Hashtags",
            header_style=f"bold {TEXT_SECONDARY}",
            style=f"bold {PRIMARY_COLOR}",
            width=first_col_width,
            no_wrap=False,
        )
        hashtag_table.add_column(
            "Description", header_style=f"bold {TEXT_SECONDARY}", style=PRIMARY_LIGHT
        )

        for tag_name in sorted(self._hashtags.keys()):
            tag_desc = self._hashtags[tag_name]
            hashtag_table.add_row(
                f"  [underline]{tag_name}[/underline]", f"  {tag_desc}"
            )

        # Group all content together
        content = RichGroup(
            Text.from_markup(
                "Examine the tables below to see "
                f"what commands, e.g. ([{SECONDARY_COLOR}]/model[/{SECONDARY_COLOR}]), "
                f"and hastags, e.g. ([{PRIMARY_COLOR}]#context[/{PRIMARY_COLOR}]), "
                "are available to you:\n",
                style=TEXT_SECONDARY,
            ),
            cmd_table,
            Text.from_markup(
                f"\n* Type [{SECONDARY_COLOR}]#[/{SECONDARY_COLOR}] to see completions\n",
                style=f"italic {TEXT_DIM}",
            ),
            hashtag_table,
            Text.from_markup(
                f"\n* Type [{PRIMARY_COLOR}]#[/{PRIMARY_COLOR}] to see completions",
                style=f"italic {TEXT_DIM}",
            ),
        )

        return content

    def _cmd_init(self) -> None:
        """Placeholder for Astro project initialization."""
        self._print_error("Command '/init' is not implemented yet.")

    def _cmd_welcome(self) -> None:
        """Render the welcome header."""
        self._print(get_welcome_header())

    def _cmd_help(self) -> None:
        """Render the help message."""
        help_display = self._build_help_text()
        self._print(help_display)

    def _cmd_exit(self) -> None:
        """Exit the application gracefully.

        Raises:
            SystemExit: Always raised to terminate the process.
        """

        self._render_rule(build_exit_rule_style())
        _loggy.debug("Checkpoint -- stopping AstroCLI")
        raise SystemExit(0)

    def _cmd_model(self) -> None:
        """Run the interactive model selection prompt."""

        try:
            selection = prompt_model_selection()
        except KeyboardInterrupt:
            return
        self._state.switch_model_to(selection.details)
        self._initialise_chat_agent()
        self._print(
            f"[{TEXT_SECONDARY}]Selected model:[/{TEXT_SECONDARY}] {selection.styled_identifier}"
        )

    async def run(self) -> None:
        """Start the interactive CLI session."""

        self._cmd_welcome()
        try:
            while True:
                # Capture user input for each prompt iteration.
                user_input = (await self._get_input(as_block=False)).strip()

                if not user_input:
                    continue

                self._start_timer()
                prompt = await self._handle_user_prompt(user_input)
                current_timestamp = get_datetime_now(to_local=True)
                self._render_rule(build_response_start_rule_style(current_timestamp))

                if prompt.startswith("/"):
                    # Lookup and execute registered slash commands.
                    command_name = prompt.split()[0]
                    spec = self._command_map.get(command_name)
                    if spec is None:
                        self._print_error(
                            f"Unknown command '{command_name}'. Use '/help' to see list of commands."
                        )
                    else:
                        try:
                            result = spec.handler()
                            if inspect.isawaitable(result):
                                await result
                        except Exception as error:  # pragma: no cover - defensive guard
                            self._print_error(
                                f"Error executing {command_name}: {error}"
                            )
                else:
                    await self._stream_chat_response(prompt)

                elapsed = self._stop_timer()
                # self._print_timestamp(elapsed)
                self._render_rule(build_response_end_rule_style(elapsed))
                self._previous_response_duration = elapsed
                self._state.save()

        except KeyboardInterrupt:
            self._cmd_exit()

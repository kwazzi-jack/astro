"""CLI entrypoint for the Astro interactive shell."""

# --- Internal Imports ---
from collections.abc import Callable, Generator
from typing import Any, Literal

# --- External Imports ---
import docstring_parser
from prompt_toolkit import HTML, PromptSession, prompt
from prompt_toolkit.application import Application, get_app
from prompt_toolkit.completion import FuzzyWordCompleter
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.lexers import PygmentsLexer
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.shortcuts import CompleteStyle, choice
from prompt_toolkit.validation import ValidationError, Validator
from prompt_toolkit.widgets import Box, Button, CheckboxList, Frame, Label, TextArea
from pydantic import BaseModel
from rich.console import Console, RenderableType
from rich.markdown import Markdown
from rich.style import Style

# --- Local Imports ---
import astro
from astro.app.state import _AppState
from astro.llms.base import KnownModels, ModelDetails
from astro.loggings.base import get_loggy
from astro.meta import get_local_package_version
from astro.paths import APPSTATE_PATH
from astro.theme import (
    BORDER_COLOR,
    ERROR_RED,
    PRIMARY_COLOR,
    TEXT_DIM,
    TEXT_SECONDARY,
    MultiCompleter,
    create_model_options,
    get_welcome_header,
    html_obj_to_rich_format,
    make_hashtag_completer,
    make_model_search_style,
    make_prompt_style,
    make_slash_command_completer,
    make_slash_lexer,
)
from astro.utilities.timing import (
    create_timer,
    get_date_str,
    get_datetime_now,
    get_time_str,
    seconds_to_strtime,
)

# --- GLOBALS ---
_loggy = get_loggy(__file__)

# Prompt components
_placeholder = HTML(f'<style fg="{TEXT_DIM}"><i>Enter a message...</i></style>')


# --- Prompt Setup ---
def _create_bottom_toolbar_function(state: _AppState) -> Callable[[], str]:
    """Render the bottom toolbar displaying the current timestamp."""

    def create_bottom_toolbar() -> str:
        current_model = state.current_model.to_identifier()
        dt = get_datetime_now(to_local=True)
        timestamp = get_time_str("%H:%M:%S (%Z)", dt)
        datestamp = get_date_str(dt=dt)
        width = get_app().output.get_size().columns
        left = f"  {timestamp}, {datestamp}"
        right = f" {current_model} ⌬ "
        space = " " * (width - len(left) - len(right) - 2)
        return left + space + right

    return create_bottom_toolbar


def _create_prompt_prefix_function(state: _AppState) -> Callable[[], HTML]:
    """Create function to render the prompt prefix to display execution count"""

    def create_prompt_prefix() -> HTML:
        return HTML(
            f'<style fg="{TEXT_SECONDARY}">[</style>{state.exec_count}<style fg="{TEXT_SECONDARY}">]</style> > '
        )

    return create_prompt_prefix


def _get_docstring_short_description(obj: Any) -> str:
    """Return the short description extracted from an object's docstring.

    Args:
        obj: Object to inspect.

    Returns:
        The short description parsed from the docstring.

    Raises:
        ValueError: If the object has no docstring or no short description.
    """
    if not hasattr(obj, "__doc__"):
        raise _loggy.ValueError(f"Cannot get docstring from {obj!r} with no docstring")
    result = docstring_parser.parse_from_object(obj).short_description
    if result is None:
        raise _loggy.ValueError(f"Could not find short-description for {obj!r}")
    return result


class _SelectedModel(BaseModel):
    chosen_identifier: str | None = None
    styled_identifier: str | None = None

    def to_model_details(self) -> ModelDetails:
        if self.chosen_identifier is None:
            raise _loggy.CreationError(
                object_type=ModelDetails, reason="chosen_identifier is None"
            )
        return KnownModels.parse(self.chosen_identifier)


def _run_model_search() -> _SelectedModel:
    result = _SelectedModel()

    # create_model_options() -> list[tuple[id:str, label:str]]
    pairs = create_model_options()
    model_map: dict[str, HTML] = dict(pairs)

    words = list(model_map.keys())
    completer = FuzzyWordCompleter(words)

    class InOptions(Validator):
        def validate(self, document):
            text = document.text.strip()
            if text in model_map:
                return
            raise ValidationError(
                message="Choose a valid option identifier.",
                cursor_position=len(text),
            )

    result.chosen_identifier = prompt(
        "Search model: ",
        completer=completer,
        complete_while_typing=True,
        complete_in_thread=True,
        validator=InOptions(),
        # style=make_model_search_style(),
    ).strip()

    result.styled_identifier = html_obj_to_rich_format(
        model_map[result.chosen_identifier]
    )

    return result


# -- CLI ---
class Astro:
    """Interactive shell for Astro commands."""

    def __init__(self, overwrite_state: bool = False):
        # Load app state
        self._state = self._load_appstate(overwrite_state)

        self._console = Console()

        # Slash commands
        self._commands: dict[str, Callable[..., Any]] = {
            "/init": self._cmd_init,
            "/help": self._cmd_help,
            "/quit": self._cmd_exit,
            "/welcome": self._cmd_welcome,
            "/model": self._cmd_model,
        }

        # Hashtags
        self._hashtags: dict[str, str] = {
            "#context": "Reference conversation context",
            "#history": "Reference chat history",
            "#system": "Reference system information",
        }

        # Create completers with styling
        command_meta_dict = {
            key: _get_docstring_short_description(func)
            for key, func in self._commands.items()
        }
        command_completer = make_slash_command_completer(command_meta_dict)
        hashtag_completer = make_hashtag_completer(self._hashtags)

        multi_completer = MultiCompleter(command_completer, hashtag_completer)

        # Create lexer with both command and hashtag support
        slash_lexer_cls = make_slash_lexer(self._commands.keys(), self._hashtags.keys())
        self._exec_count = -1
        self._session = PromptSession(
            message=_create_prompt_prefix_function(self._state),
            placeholder=_placeholder,
            bottom_toolbar=_create_bottom_toolbar_function(self._state),
            color_depth=ColorDepth.TRUE_COLOR,
            completer=multi_completer,
            complete_style=CompleteStyle.COLUMN,
            complete_in_thread=True,
            complete_while_typing=True,
            lexer=PygmentsLexer(slash_lexer_cls),
            style=make_prompt_style(),
            refresh_interval=0.1,
            wrap_lines=True,
            cursor=CursorShape.BLINKING_BLOCK,
        )
        start, stop = create_timer()
        self._start_func = start
        self._stop_func = stop

    def _load_appstate(self, overwrite: bool = False) -> _AppState:
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
            except Exception as error:
                raise _loggy.CreationError(
                    object_type=_AppState,
                    reason="Tried to create new file",
                    caused_by=error,
                )
        except Exception as error:
            raise _loggy.LoadError(
                path_or_uid=APPSTATE_PATH,
                obj_or_key=_AppState,
                load_from="json",
                caused_by=error,
            )

    def _start_timer(self) -> None:
        return self._start_func()

    def _stop_timer(self) -> float:
        return self._stop_func(True)

    def _print_newline(self) -> None:
        """Print a newline."""
        self._console.print()

    def _get_input(self, as_block: bool = True) -> str:
        """Prompt the user for input."""
        if as_block:
            self._print_newline()
        result = self._session.prompt()
        if as_block:
            self._print_newline()
        return result

    def _print(self, obj: RenderableType, as_block: bool = True) -> None:
        """Print a renderable object with optional empty lines around it."""
        if as_block:
            self._print_newline()
        self._console.print(obj)
        if as_block:
            self._print_newline()

    def _print_error(self, message: RenderableType) -> None:
        """Print an error message with standard formatting."""
        self._print(
            f"[bold {ERROR_RED}]>>> ERROR:[/bold {ERROR_RED}] [{ERROR_RED}]{message}[/{ERROR_RED}] [bold {ERROR_RED}]<<<[/bold {ERROR_RED}]"
        )

    def _draw_rule(
        self,
        text: str = "",
        newline: bool = False,
        character: str = "─",
        style: str | Style | None = None,
        align: Literal["left", "center", "right"] = "center",
    ) -> None:
        """Draw a horizontal rule in the console."""
        if style is None:
            style = BORDER_COLOR
        self._console.rule(text, characters=character, align=align, style=style)
        if newline:
            self._print_newline()

    def _draw_exit_rule(self) -> None:
        """Draw a horizontal rule for exit."""
        self._draw_rule(
            f"[{ERROR_RED}]EXIT[/{ERROR_RED}]", character="━", style=ERROR_RED
        )

    def _draw_response_start_rule(self) -> None:
        timestamp = get_time_str("%H:%M:%S.%f")[:-3]
        self._draw_rule(f"[{TEXT_DIM}]{timestamp}[/{TEXT_DIM}]", character="╌")

    def _draw_response_end_rule(self, time_taken: float) -> None:
        """Draw a rule showing the response time."""
        strtime = seconds_to_strtime(time_taken)
        self._draw_rule(f"[{TEXT_DIM}]{strtime}[/{TEXT_DIM}]")

    def _cmd_init(self) -> None:
        """Initializes a new Astro project."""
        # This is a placeholder for the actual implementation.
        self._print_error("Command '/init' is not implemented yet.")

    def _cmd_welcome(self) -> None:
        """Shows the welcome message"""
        self._print(get_welcome_header())

    def _cmd_help(self) -> None:
        """Shows the help message."""
        help_text = """# Astro CLI Help

Available commands:

- `/help`    - Shows this help message
- `/welcome` - Shows welcome message
- `/model`   - Change model to use
- `/init`    - Initializes a new Astro project (not yet implemented)
- `/quit`    - Exits the application

Type a message to chat with Astro, or use a command to perform actions."""
        self._print(Markdown(help_text))

    def _cmd_exit(self) -> None:
        """Exits the application."""
        self._draw_exit_rule()
        exit(0)

    def _cmd_model(self) -> None:
        """Runs the model change function"""
        try:
            result = _run_model_search()
            model_details = result.to_model_details()
            self._state.switch_model_to(model_details)
            self._print(
                f"[{TEXT_SECONDARY}]Selected model:[/{TEXT_SECONDARY}] {result.styled_identifier}"
            )
        except KeyboardInterrupt:
            return

    def run(self):
        """Start the interactive session."""
        self._cmd_welcome()
        try:
            while True:
                user_input = self._get_input().strip()
                self._state.increment_count()
                self._draw_response_start_rule()

                # Skip empty input
                if not user_input:
                    continue

                self._start_timer()
                if user_input.startswith("/"):
                    command_name = user_input.split()[0]
                    command = self._commands.get(command_name)
                    if command:
                        try:
                            # For now, not passing arguments
                            command()
                        except Exception as e:
                            self._print_error(f"Error executing {command_name}: {e}")
                    else:
                        self._print_error(
                            f"Unknown command '{command_name}'. Use '/help' to see list of commands."
                        )
                else:
                    self._print(
                        f"[bold {PRIMARY_COLOR}]Astro:[/bold {PRIMARY_COLOR}] {user_input}?"
                    )
                time_taken = self._stop_timer()
                self._draw_response_end_rule(time_taken)
                self._state.save()

        except KeyboardInterrupt:
            self._cmd_exit()

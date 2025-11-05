"""Styling, lexer, and completer utilities for the Astro CLI."""

# --- Internal Imports ---
import bisect
import itertools
import math
import random
import re
import shutil
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, TypeAlias, overload

# --- External Imports ---
from colour import Color
from prompt_toolkit import HTML, prompt
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion, FuzzyWordCompleter
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import BaseStyle, Style, merge_styles
from prompt_toolkit.styles.pygments import style_from_pygments_dict
from prompt_toolkit.validation import ValidationError, Validator
from pydantic import BaseModel
from pygments.lexer import RegexLexer
from pygments.token import Generic, Name, Text
from rich.align import Align as RichAlign
from rich.console import Group
from rich.theme import Theme
from rich.markdown import Markdown

# --- Local Imports ---
from astro.llms.base import KnownModels, ModelDetails
from astro.logger import get_loggy
from astro.meta import get_astro_version
from astro.typings import (
    DateTimeFactory,
    HTMLDict,
    HTMLFactory,
    PTKDecoration,
    StrDict,
    StringFactory,
)
from astro.utilities.display import get_terminal_width
from astro.utilities.timing import get_time_str, seconds_to_strtime

# --- GLOBALS ---
_loggy = get_loggy(__file__)


# --- Theme Helper Functions and Classes ---
def _apply_bold_html(text: str) -> str:
    """Applies the bold html tags to text."""
    return f"<b>{text}</b>"


def _apply_italic_html(text: str) -> str:
    """Applies the italics html tags to text."""
    return f"<i>{text}</i>"


def _apply_underline_html(text: str) -> str:
    """Applies the underline html tags to text."""
    return f"<u>{text}</u>"


def _apply_strikethrough_html(text: str) -> str:
    """Applies the strikethrough html tags to text."""
    return f"<s>{text}</s>"

def _apply_bold_md(text: str) -> str:
    """Applies the bold on markdown text."""
    return f"**{text}**"


def _apply_italic_md(text: str) -> str:
    """Applies the italics on markdown text."""
    return f"*{text}*"


def _apply_strikethrough_md(text: str) -> str:
    """Applies the strikethrough on markdown text."""
    return f"~~{text}~~"


def _get_ptk_app_width() -> int:
    """Gets the current width of the prompt session terminal."""
    return get_app().output.get_size().columns


class AstroStyle(BaseModel):
    fg: str | None = None
    bg: str | None = None
    bold: bool = False
    italic: bool = False
    decoration: PTKDecoration = "none"
    ignore_trailing_spaces: bool = False


def _apply_ptk_style1(
    text: str,
    *,
    fg: str | None = None,
    bg: str | None = None,
    bold: bool = False,
    italic: bool = False,
    decoration: Literal["underline", "strikethrough", "none"] = "none",
    ignore_trailing_spaces: bool = False,
) -> HTML:
    """Applies prompt_toolkit styling to the text and returns an HTML object."""
    style_str_parts = []

    # Apply foreground
    if fg is not None:
        style_str_parts.append(f"{fg=}")

    # Apply background
    if bg is not None:
        style_str_parts.append(f"{bg=}")

    # Construct style string
    style_str = " ".join(style_str_parts)

    # Whether to format trailing spaces
    if ignore_trailing_spaces:
        leading_count = len(text) - len(text.lstrip())
        trailing_count = len(text) - len(text.rstrip())
        stripped_text = text.strip()
        stripped_html_str = (
            f"<style {style_str}>{stripped_text}</style>"
            if len(style_str) > 0
            else text
        )
        html_str = " " * leading_count + stripped_html_str + " " * trailing_count
    else:
        html_str = f"<style {style_str}>{text}</style>" if len(style_str) > 0 else text

    # Apply bold weight
    if bold:
        html_str = _apply_bold_html(html_str)

    # Apply italic style
    if italic:
        html_str = _apply_italic_html(html_str)

    # Apply decoration
    match decoration:
        case "underline":
            html_str = _apply_underline_html(html_str)
        case "strikethrough":
            html_str = _apply_strikethrough_html(html_str)

    # Return as HTML object
    return HTML(html_str)


def _apply_ptk_style2(text: str, style: AstroStyle | None = None) -> HTML:
    """Applies prompt_toolkit styling to the text and returns an HTML object."""
    return _apply_ptk_style1(
        text, **style.model_dump() if style is not None else AstroStyle().model_dump()
    )


def _apply_rich_style1(
    text: str,
    *,
    fg: str | None = None,
    bg: str | None = None,
    bold: bool = False,
    italic: bool = False,
    decoration: Literal["underline", "strikethrough", "none"] = "none",
    ignore_trailing_spaces: bool = False,
) -> str:
    """Applies Rich markup styling to the text and returns a formatted string."""
    # Escape the text first to prevent Rich markup interpretation
    escaped_text = escape_rich_markup(text)

    # Whether to format trailing spaces
    if ignore_trailing_spaces:
        leading_count = len(text) - len(text.lstrip())
        trailing_count = len(text) - len(text.rstrip())
        stripped_text = text.strip()
        escaped_text = escape_rich_markup(stripped_text)
        result = " " * leading_count + escaped_text + " " * trailing_count
    else:
        result = escaped_text

    # Build style parts
    style_parts = []

    # Apply bold
    if bold:
        style_parts.append("bold")

    # Apply italic
    if italic:
        style_parts.append("italic")

    # Apply decoration
    if decoration == "underline":
        style_parts.append("underline")
    elif decoration == "strikethrough":
        style_parts.append("strike")

    # Apply foreground
    if fg is not None:
        style_parts.append(fg)

    # Apply background
    if bg is not None:
        style_parts.append(f"on {bg}")

    # Construct final markup
    if style_parts:
        style_str = " ".join(style_parts)
        return f"[{style_str}]{result}[/{style_str}]"
    else:
        return result


def _apply_rich_style2(text: str, style: AstroStyle | None = None) -> str:
    """Applies Rich markup styling to the text and returns a formatted string."""
    return _apply_rich_style1(
        text, **style.model_dump() if style is not None else AstroStyle().model_dump()
    )

def _apply_markdown_style1(text: str, style: AstroStyle | None = None) -> Markdown:


def join_html_obj(*objs: HTML | str, separator: str = "") -> HTML:
    """Joins multiple HTML objects into a single HTML object."""
    return HTML(
        separator.join(obj.value if isinstance(obj, HTML) else obj for obj in objs)
    )


@dataclass(frozen=True)
class PromptConfiguration:
    """Configuration container for constructing the interactive prompt session.

    Attributes:
        prompt_prefix (HTMLFactory): Callable that returns the prefix
            markup before user input.
        placeholder (HTML): Placeholder markup displayed when the input is
            empty.
        bottom_toolbar (StringFactory): Callable that renders the toolbar
            content at the bottom.
        completer (Completer): Prompt_toolkit completer configured for commands
            and hashtags.
        lexer_cls (type[RegexLexer]): Lexer class used for inline syntax
            highlighting.
        style (BaseStyle): Prompt_toolkit style merging pygments and UI
            selectors.
        color_depth (ColorDepth): Desired output color depth for the prompt
            session. Defaults to ColorDepth.TRUE_COLOR.
        complete_style (CompleteStyle): Completion menu arrangement strategy.
            Defaults to CompleteStyle.COLUMN.
        complete_in_thread (bool): Whether completion resolution should occur in
            a background thread. Defaults to True.
        complete_while_typing (bool): Whether completions should appear while
            the user types. Defaults to True.
        validator (Validator | None): Optional validator applied to prompt
            input. Defaults to None.
        validate_while_typing (bool): Whether validation feedback should appear
            while typing. Defaults to False.
        refresh_interval (float): UI refresh interval for dynamic components in
            seconds. Defaults to 0.1.
        wrap_lines (bool): Whether lines should wrap within the prompt.
            Defaults to True.
        cursor_shape (CursorShape): Cursor shape displayed within the prompt.
            Defaults to CursorShape.BLINKING_BLOCK.
    """

    prompt_prefix: HTMLFactory
    placeholder: HTML
    bottom_toolbar: StringFactory
    completer: Completer
    lexer_cls: type[RegexLexer]
    style: BaseStyle
    color_depth: ColorDepth = ColorDepth.TRUE_COLOR
    complete_style: CompleteStyle = CompleteStyle.COLUMN
    complete_in_thread: bool = True
    complete_while_typing: bool = True
    validator: Validator | None = None
    validate_while_typing: bool = False
    refresh_interval: float = 0.1
    wrap_lines: bool = True
    cursor_shape: CursorShape = CursorShape.BLINKING_BLOCK


@dataclass(frozen=True)
class RuleStyle:
    """Describe the rendering details for Rich console rules.

    Attributes:
        text (str): Label rendered at the center of the rule.
            Defaults to an empty string.
        character (str): Character used to draw the horizontal line.
            Defaults to "─".
        style (str | Style | None): Optional Rich style expression or Style
            instance applied to the rule. Defaults to None.
        align (Literal['left', 'center', 'right']): Alignment of the label
            within the rule. Defaults to "center".
        newline (bool): Whether to print a trailing newline after the rule.
            Defaults to False.
    """

    text: str = ""
    character: str = "─"
    style: str | Style | None = None
    align: Literal["left", "center", "right"] = "center"
    newline: bool = False


@dataclass(frozen=True)
class ModelSelectionResult:
    """Container describing the outcome of the model selection prompt.

    Attributes:
        identifier (str): Identifier chosen by the user.
        styled_identifier (str): Rich markup for the identifier presentation.
        details (ModelDetails): Parsed model details for downstream usage.
    """

    identifier: str
    styled_identifier: str
    details: ModelDetails


def create_ptk_prompt_prefix(
    name: str,
    symbol: str = "⟩",
    name_style: AstroStyle | None = None,
    symbol_style: AstroStyle | None = None,
    append_spaces: int | None = None,
) -> HTML:
    name_html = _apply_ptk_style2(name, style=name_style)
    symbol_html = _apply_ptk_style2(symbol, style=symbol_style)
    if append_spaces is not None and append_spaces >= 0:
        symbol_html = join_html_obj(symbol_html, " " * append_spaces)
    return join_html_obj(
        name_html,
        symbol_html,
        separator=" ",
    )


def create_ptk_prompt_prefix_factory(
    name_or_provider: str | StringFactory,
    symbol_or_provider: str | StringFactory = "⟩",
    name_style: AstroStyle | None = None,
    symbol_style: AstroStyle | None = None,
    append_spaces: int | None = None,
) -> HTMLFactory:
    # Normalize providers to callable functions
    if callable(name_or_provider):
        name_provider = name_or_provider
    else:

        def name_provider() -> str:
            return name_or_provider

    if callable(symbol_or_provider):
        symbol_provider = symbol_or_provider
    else:

        def symbol_provider() -> str:
            return symbol_or_provider

    def create_prompt_inner() -> HTML:
        return create_ptk_prompt_prefix(
            name_provider(),
            symbol_provider(),
            name_style=name_style,
            symbol_style=symbol_style,
            append_spaces=append_spaces,
        )

    return create_prompt_inner


def create_rich_prompt_prefix(
    name: str,
    symbol: str = "⟩",
    name_style: AstroStyle | None = None,
    symbol_style: AstroStyle | None = None,
    append_spaces: int | None = None,
) -> str:
    name_markup = _apply_rich_style2(name, style=name_style)
    symbol_markup = _apply_rich_style2(symbol, style=symbol_style)
    return (
        name_markup + symbol_markup + " " * append_spaces
        if append_spaces is not None and append_spaces >= 0
        else ""
    )


def create_user_prompt_style() -> AstroStyle:
    return AstroStyle(fg=SECONDARY_COLOR)


def create_astro_prompt_style() -> AstroStyle:
    return AstroStyle(fg=PRIMARY_COLOR)


def create_system_prompt_style() -> AstroStyle:
    return AstroStyle(fg=TEXT_SECONDARY)


def create_bottom_toolbar_factory(
    model_identifier_provider: StringFactory,
    datetime_provider: DateTimeFactory,
) -> StringFactory:
    """Create a callable that renders the bottom toolbar with lightweight metadata.

    Args:
        model_identifier_provider (StringFactory): Provider returning the
            active model identifier string.
        datetime_provider (DateTimeFactory): Provider returning the
            current datetime value.

    Returns:
        StringFactory: Callable producing the toolbar string for
        prompt_toolkit.
    """

    # Create toolbar string factory function
    def draw_toolbar() -> str:
        # Select current model
        current_model = model_identifier_provider()
        left = f" {current_model}"

        # Get current datetime and format
        dt_value = datetime_provider()
        timestamp = get_time_str("%H:%M:%S", dt_value)
        right = f" {timestamp} "

        # Get width and calculate spacing
        width = _get_ptk_app_width()
        space_count = max(0, width - len(left) - len(right))
        spacing = " " * space_count

        # Return padded format string
        return left + spacing + right

    return draw_toolbar


def create_prompt_configuration(
    username_provider: StringFactory,
    model_identifier_provider: StringFactory,
    datetime_provider: DateTimeFactory,
    command_metadata: StrDict,
    hashtag_metadata: StrDict,
) -> PromptConfiguration:
    """Create the prompt configuration for the interactive CLI session.

    Args:
        username_provider (StringFactory): Provider returning the active
            username.
        model_identifier_provider (StringFactory): Provider returning the
            active model identifier.
        datetime_provider (DateTimeFactory): Provider returning the
            current datetime.
        command_metadata (StrDict): Mapping of command names to their
            descriptions.
        hashtag_metadata (StrDict): Mapping of hashtag trigger strings to
            their descriptions.

    Returns:
        PromptConfiguration: Configuration object tailored for the Astro CLI
        prompt session.
    """
    name_style = create_user_prompt_style()
    symbol_style = create_system_prompt_style()

    prompt_prefix = create_ptk_prompt_prefix_factory(
        username_provider,
        name_style=name_style,
        symbol_style=symbol_style,
        append_spaces=1,
    )
    bottom_toolbar = create_bottom_toolbar_factory(
        model_identifier_provider, datetime_provider
    )
    placeholder = HTML(f'<style fg="{TEXT_DIM}"><i>Enter a message...</i></style>')

    command_completer = make_slash_command_completer(command_metadata)
    hashtag_completer = make_hashtag_completer(hashtag_metadata)
    combined_completer = MultiCompleter(command_completer, hashtag_completer)
    lexer_cls = make_slash_lexer(command_metadata.keys(), hashtag_metadata.keys())
    style = make_prompt_style()
    validator = _CommandInputValidator(
        known_commands=set(command_metadata.keys()),
        known_hashtags=set(hashtag_metadata.keys()),
    )

    return PromptConfiguration(
        prompt_prefix=prompt_prefix,
        placeholder=placeholder,
        bottom_toolbar=bottom_toolbar,
        completer=combined_completer,
        lexer_cls=lexer_cls,
        style=style,
        validator=validator,
        validate_while_typing=True,
    )


def build_error_banner(message: str) -> str:
    """Create a standardized error banner for console rendering.

    Args:
        message (str): Error message to highlight.

    Returns:
        str: Rich markup string representing the error banner.
    """

    return (
        f"[bold {ERROR_RED}]>>> ERROR:[/bold {ERROR_RED}] "
        f"[{ERROR_RED}]{message}[/{ERROR_RED}] "
        f"[bold {ERROR_RED}]<<<[/bold {ERROR_RED}]"
    )


def build_rule_style(
    text: str = "",
    *,
    character: str = "─",
    style: str | Style | None = None,
    align: Literal["left", "center", "right"] = "center",
    newline: bool = False,
) -> RuleStyle:
    """Create a rule style descriptor for repeated usage patterns.

    Args:
        text (str): Label rendered at the center of the rule. Defaults to an
            empty string.
        character (str): Character used for drawing the horizontal rule.
            Defaults to "─".
        style (str | Style | None): Optional Rich style expression or Style
            instance for the rule. Defaults to None.
        align (Literal['left', 'center', 'right']): Alignment of the rule text.
            Defaults to "center".
        newline (bool): Whether a newline should follow the rule. Defaults to
            False.

    Returns:
        RuleStyle: Descriptor describing the rendering parameters.
    """

    return RuleStyle(
        text=text,
        character=character,
        style=style,
        align=align,
        newline=newline,
    )


def build_exit_rule_style() -> RuleStyle:
    """Create the rule style used when the CLI exits.

    Returns:
        RuleStyle: Descriptor for the exit rule styling.
    """

    return build_rule_style(
        text=f"[{ERROR_RED}]EXIT[/{ERROR_RED}]",
        character="━",
        style=ERROR_RED,
    )


def build_response_start_rule_style(timestamp: datetime) -> RuleStyle:
    """Create the rule style displayed before rendering a model response.

    Args:
        timestamp (datetime): Datetime value marking when the response
            processing began.

    Returns:
        RuleStyle: Descriptor for the response start divider styling.
    """

    formatted_timestamp = get_time_str("%H:%M:%S.%f", timestamp)[:-3]
    return build_rule_style(
        text=f"[{TEXT_DIM}]{formatted_timestamp}[/{TEXT_DIM}]",
        character="╌",
        style=BORDER_COLOR,
    )


def build_response_end_rule_style(duration_seconds: float) -> RuleStyle:
    """Create the rule style displayed after rendering a model response.

    Args:
        duration_seconds (float): Duration in seconds for the completed
            response.

    Returns:
        RuleStyle: Descriptor for the response end divider styling.
    """

    readable = seconds_to_strtime(duration_seconds)
    return build_rule_style(
        text=f"[{TEXT_DIM}]{readable}[/{TEXT_DIM}]",
        character="─",
        style=BORDER_COLOR,
    )


class _ModelOptionValidator(Validator):
    """Ensure a chosen identifier is present in the available model map."""

    def __init__(self, valid_options: set[str]):
        self._valid_options = valid_options

    def validate(self, document) -> None:  # noqa: D401 - prompt toolkit signature
        text = document.text.strip()
        if text in self._valid_options:
            return
        raise ValidationError(
            message="Choose a valid option identifier.",
            cursor_position=len(text),
        )


def prompt_model_selection() -> ModelSelectionResult:
    """Prompt the user to choose an available model identifier.

    Returns:
        ModelSelectionResult: Result describing the chosen identifier and
        associated details.

    Raises:
        ValueError: Raised if parsing the identifier or converting to
            ModelDetails fails.
    """

    # Get model selections to show
    entries: HTMLDict = dict(create_model_options())

    # Create prompt session components
    completer = FuzzyWordCompleter(list(entries.keys()))
    validator = _ModelOptionValidator(set(entries.keys()))
    _prompt_text = _apply_ptk_style1(
        "Search model: ",
        fg=TEXT_SECONDARY,
        decoration="underline",
        ignore_trailing_spaces=True,
    )

    # Create prompt session
    chosen_identifier = prompt(
        _prompt_text,
        completer=completer,
        complete_while_typing=True,
        complete_in_thread=True,
        validator=validator,
        style=make_model_search_style(),
    ).strip()

    # Get HTML markup of object and convert
    html_markup = entries[chosen_identifier]
    styled_identifier = html_obj_to_rich_format(html_markup)

    # Try to parse identifier
    try:
        details = KnownModels.parse(chosen_identifier)

    # Something went wrong
    except Exception as error:
        raise _loggy.CreationError(
            object_type=ModelDetails,
            reason=f"Identifier {chosen_identifier!r} could not be parsed",
            caused_by=error,
        )

    # Return result object of selected model
    return ModelSelectionResult(
        identifier=chosen_identifier,
        styled_identifier=styled_identifier,
        details=details,
    )


def _centre_line_to_terminal(line: str, width: int) -> str:
    """Centre a single line within the given terminal width.

    Args:
        line (str): Line content to centre.
        width (int): Available terminal width in characters.

    Returns:
        str: Centred line padded with spaces.
    """

    line_length = len(line)
    padding = (width - line_length) // 2
    return " " * padding + line


def _centre_text_to_terminal(text: str) -> str:
    """Centre multi-line text relative to the current terminal width.

    Args:
        text (str): Multi-line text that should be centred.

    Returns:
        str: Centred text joined with newline characters.
    """

    centred_lines = []
    width = get_terminal_width()
    for line in text.splitlines():
        centred_lines.append(_centre_line_to_terminal(line, width))
    return "\n".join(centred_lines)


def _generate_colour_luminance_variation(base_hex: str, scale: float) -> str:
    """Generate a color variation by scaling luminance within bounds.

    Creates a new color by multiplying the base color's luminance by a scale factor
    and clamping the result between minimum and maximum values.

    Args:
        base_hex (str): Base color in hex format (e.g., "#f5c75f").
        scale (float): Multiplier for luminance adjustment.

    Returns:
        str: Hex color string with adjusted luminance.
    """
    new_color = Color(base_hex)
    new_luminance = new_color.get_luminance() * scale
    new_color.set_luminance(new_luminance)
    return new_color.get_hex_l()


def _create_gradient(start_hex: str, end_hex: str, steps: int) -> list[str]:
    """Create a gradient between two colors.

    Args:
        start_hex (str): Starting color in hex format (for example "#8b5cf6").
        end_hex (str): Ending color in hex format (for example "#a78bfa").
        steps (int): Number of steps within the gradient range.

    Returns:
        list[str]: Hex color strings representing the gradient.
    """
    start = Color(start_hex)
    colors = list(start.range_to(Color(end_hex), steps))
    return [color.hex_l for color in colors]


def _make_star_sampler(weights: dict[str, int]) -> Callable[[random.Random], str]:
    """Return a weighted sampler over width-1 star glyphs.

    Args:
        weights (dict[str, int]): Mapping from glyph symbol to relative weight.

    Returns:
        Callable[[random.Random], str]: Callable that produces weighted glyphs
        using the provided random generator.
    """
    weight_items: list[tuple[str, int]] = [
        (symbol, weight) for symbol, weight in weights.items() if weight > 0
    ]
    if not weight_items:
        weight_items = [(".", 1)]

    symbols: tuple[str, ...]
    raw_weights: tuple[int, ...]
    symbols, raw_weights = zip(*weight_items)
    total_weight: float = float(sum(raw_weights))
    cumulative_distribution: list[float] = list(
        itertools.accumulate(w / total_weight for w in raw_weights)
    )

    def pick_star(random_generator: random.Random) -> str:
        roll: float = random_generator.random()  # in [0, 1)
        index: int = bisect.bisect_left(cumulative_distribution, roll)
        return symbols[index]

    return pick_star


# Weighted width-1 star glyphs
_pick_star = _make_star_sampler({".": 5, "·": 1, "˙": 5, "•": 1})


def _poisson_sample(lam: float, random_generator: random.Random) -> int:
    """Sample from Poisson(lam) with Knuth for small λ, normal approx for large λ.

    Args:
        lam (float): Rate parameter for the Poisson distribution.
        random_generator (random.Random): Random generator used for sampling.

    Returns:
        int: Sampled event count.
    """
    if lam <= 0:
        return 0
    if lam < 30:
        k: int = 0
        p: float = 1.0
        threshold: float = math.exp(-lam)
        while p > threshold:
            k += 1
            p *= random_generator.random()
        return k - 1
    return max(0, int(random_generator.normalvariate(lam, math.sqrt(lam)) + 0.5))


def _generate_starfield_2d_clustered(
    rows: int,
    columns: int,
    base_density: float,  # uniform background probability per cell (e.g., 0.02)
    cluster_rate_per_1000: float,  # expected clusters per 1000 cells (e.g., 1.2)
    cluster_mean_size: float,  # mean stars per cluster (e.g., 8)
    cluster_row_std_cells: float,  # vertical spread in cells (e.g., 2.0)
    cluster_col_std_cells: float,  # horizontal spread in cells (e.g., 4.0)
    random_generator: random.Random,
    pick_star: Callable[[random.Random], str] = _pick_star,
) -> list[list[str]]:
    """Generate a clustered 2D starfield as a grid of glyphs.

    Args:
        rows (int): Number of rows in the generated grid.
        columns (int): Number of columns in the generated grid.
        base_density (float): Uniform background probability per cell.
        cluster_rate_per_1000 (float): Expected clusters per thousand cells.
        cluster_mean_size (float): Mean number of stars per cluster.
        cluster_row_std_cells (float): Vertical spread of each cluster in cells.
        cluster_col_std_cells (float): Horizontal spread of each cluster in
            cells.
        random_generator (random.Random): Random generator used for sampling.
        pick_star (Callable[[random.Random], str]): Glyph sampler used when a
            star should be rendered. Defaults to _pick_star.

    Returns:
        list[list[str]]: Matrix of glyphs representing the starfield.
    """
    # Background layer
    grid: list[list[str]] = [
        [
            "*" if random_generator.random() < base_density else " "
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]

    # Number of clusters ~ Poisson(cluster_rate_per_1000 * (rows * columns / 1000))
    lambda_grid: float = max(0.0, cluster_rate_per_1000 * (rows * columns / 1000.0))
    number_of_clusters: int = _poisson_sample(lambda_grid, random_generator)

    # Place clusters
    for _ in range(number_of_clusters):
        center_row: int = random_generator.randrange(rows)
        center_col: int = random_generator.randrange(columns)
        offspring_count: int = max(
            1, _poisson_sample(cluster_mean_size, random_generator)
        )

        for _ in range(offspring_count):
            star_row: int = int(
                random_generator.normalvariate(center_row, cluster_row_std_cells)
            )
            star_col: int = int(
                random_generator.normalvariate(center_col, cluster_col_std_cells)
            )
            if 0 <= star_row < rows and 0 <= star_col < columns:
                grid[star_row][star_col] = "*"

    # Render marks to weighted glyphs
    for r in range(rows):
        for c in range(columns):
            if grid[r][c] == "*":
                grid[r][c] = pick_star(random_generator)
            else:
                grid[r][c] = " "
    return grid


def _squarify_text(text: str) -> str:
    """Pad each line of text to the width of the longest line.

    Args:
        text (str): Multi-line text to square.

    Returns:
        str: Text with each line centred to the maximum width.
    """

    lines = text.splitlines()
    max_length = max(len(line) for line in lines) if lines else 0
    squarified_lines = [line.center(max_length) for line in lines]
    return "\n".join(squarified_lines)


def _overlay_banner_on_starfield(
    banner: str,
    star_grid: list[list[str]],
) -> str:
    """Overlay a banner into the centre of the star grid.

    Args:
        banner (str): Multi-line banner text.
        star_grid (list[list[str]]): Existing star grid that should receive the
            banner overlay.

    Returns:
        str: Rendered banner blended with the star grid.
    """
    square_banner = _squarify_text(banner)
    banner_lines = square_banner.splitlines()
    if not banner_lines:
        return "\n".join("".join(row) for row in star_grid)

    # Assume equal length lines and no leading/trailing spaces as requested
    banner_width = len(banner_lines[1])

    # banner_width: int = len(banner_lines[0])
    rows: int = len(star_grid)
    columns: int = len(star_grid[0]) if rows else 0

    # Horizontal centering
    padding_cells: int = columns - banner_width
    if padding_cells < 0:
        # Truncate banner to available width
        banner_lines = [line[:columns] for line in banner_lines]
        banner_width = columns
        padding_cells = 0
    left_padding: int = padding_cells // 2

    # Vertical sizing: fit banner lines into grid rows
    usable_rows: int = min(rows, len(banner_lines))
    start_row: int = (rows - usable_rows) // 2  # center vertically if grid is taller

    for i in range(usable_rows):
        line_text = banner_lines[i]
        grid_row_index = start_row + i
        for j in range(banner_width):
            ch = line_text[j]
            if ch != " ":
                star_grid[grid_row_index][left_padding + j] = ch
    return "\n".join("".join(row) for row in star_grid)


def _render_banner_with_starfield(
    banner: str,
    width: int | None = None,
    seed: int | None = None,
    # Starfield controls:
    base_density: float = 0.15,
    cluster_rate_per_1000: float = 4.0,
    cluster_mean_size: float = 5.0,
    cluster_row_std_cells: float = 1.0,
    cluster_col_std_cells: float = 2.0,
    pick_star: Callable[[random.Random], str] = _pick_star,
) -> str:
    """Build a 2D clustered starfield and overlay the banner centred within it.

    Args:
        banner (str): Banner text to render.
        width (int | None): Optional explicit width for the starfield grid.
            Defaults to the detected terminal width when None.
        seed (int | None): Optional seed used to initialise the random
            generator. Defaults to None.
        base_density (float): Background probability for star placement.
            Defaults to 0.15.
        cluster_rate_per_1000 (float): Expected clusters per thousand cells.
            Defaults to 4.0.
        cluster_mean_size (float): Mean number of stars per cluster. Defaults to
            5.0.
        cluster_row_std_cells (float): Vertical spread for cluster generation in
            cells. Defaults to 1.0.
        cluster_col_std_cells (float): Horizontal spread for cluster generation
            in cells. Defaults to 2.0.
        pick_star (Callable[[random.Random], str]): Glyph sampler used for star
            placement. Defaults to _pick_star.

    Returns:
        str: Rendered banner embedded within the generated starfield.
    """
    terminal_width: int = shutil.get_terminal_size().columns if width is None else width
    banner_lines: list[str] = banner.splitlines()
    rows = len(banner_lines) if banner_lines else 0
    if rows == 0:
        return ""

    random_generator = random.Random(seed)
    star_grid: list[list[str]] = _generate_starfield_2d_clustered(
        rows=rows,
        columns=terminal_width,
        base_density=base_density,
        cluster_rate_per_1000=cluster_rate_per_1000,
        cluster_mean_size=cluster_mean_size,
        cluster_row_std_cells=cluster_row_std_cells,
        cluster_col_std_cells=cluster_col_std_cells,
        random_generator=random_generator,
        pick_star=pick_star,
    )
    return _overlay_banner_on_starfield(banner=banner, star_grid=star_grid)


# --- Palette ---
# Core brand
PRIMARY_COLOR = "#d06cc7"
PRIMARY_LIGHT = _generate_colour_luminance_variation(PRIMARY_COLOR, scale=1.2)
PRIMARY_DARK = _generate_colour_luminance_variation(PRIMARY_COLOR, scale=0.5)
PRIMARY_DIM = _generate_colour_luminance_variation(PRIMARY_COLOR, scale=0.2)

SECONDARY_COLOR = "#6c97d0"
SECONDARY_LIGHT = _generate_colour_luminance_variation(SECONDARY_COLOR, scale=1.2)
SECONDARY_DARK = _generate_colour_luminance_variation(SECONDARY_COLOR, scale=0.5)
SECONDARY_DIM = _generate_colour_luminance_variation(SECONDARY_COLOR, scale=0.2)


# Neutrals
TEXT_PRIMARY = "#e2e8f0"  # primary text (light slate)
TEXT_SECONDARY = "#94a3b8"  # secondary/muted text (slate)
TEXT_DIM = "#64748b"  # dim text
BG_PRIMARY = "#0f172a"  # primary background (dark slate)
BG_SECONDARY = "#1e293b"  # secondary background (elevated)
BORDER_COLOR = "#334155"  # border color


# Semantic colors
SUCCESS_GREEN = "#10b981"  # emerald
INFO_BLUE = "#3b82f6"  # blue
WARNING_AMBER = "#f59e0b"  # amber
ERROR_RED = "#ef4444"  # red
ACCENT_CYAN = "#06b6d4"  # cyan
ACCENT_GOLD = "#fbbf24"  # gold


# UI elements
CURSOR_COLOR = "#c4b5fd"  # light purple tint
SELECTION_BG = "#312e81"  # purple selection


def make_rich_theme() -> Theme:
    """Create a Rich Theme object matching the prompt_toolkit color palette.

    Returns:
        Theme: Rich theme configured with Astro color palette.
    """
    return Theme(
        {
            # Brand colors
            "primary": PRIMARY_COLOR,
            "primary.light": PRIMARY_LIGHT,
            "primary.dark": PRIMARY_DARK,
            "primary.dim": PRIMARY_DIM,
            "secondary": SECONDARY_COLOR,
            "secondary.light": SECONDARY_LIGHT,
            "secondary.dark": SECONDARY_DARK,
            "secondary.dim": SECONDARY_DIM,
            # Text colors
            "text.primary": TEXT_PRIMARY,
            "text.secondary": TEXT_SECONDARY,
            "text.dim": TEXT_DIM,
            # Background colors
            "bg.primary": f"on {BG_PRIMARY}",
            "bg.secondary": f"on {BG_SECONDARY}",
            "border": BORDER_COLOR,
            # Semantic colors
            "success": SUCCESS_GREEN,
            "info": INFO_BLUE,
            "warning": WARNING_AMBER,
            "error": ERROR_RED,
            "accent.cyan": ACCENT_CYAN,
            "accent.gold": ACCENT_GOLD,
            # Markdown-specific overrides to match prompt_toolkit theme
            "markdown.h1": f"bold {SECONDARY_COLOR}",
            "markdown.h2": f"bold {PRIMARY_COLOR}",
            "markdown.h3": f"bold {PRIMARY_LIGHT}",
            "markdown.h4": f"bold {TEXT_PRIMARY}",
            "markdown.h5": f"bold {TEXT_SECONDARY}",
            "markdown.h6": f"bold {TEXT_DIM}",
            "markdown.code": f"{ACCENT_CYAN}",
            "markdown.code_block": f"{TEXT_PRIMARY} on {BG_SECONDARY}",
            "markdown.link": f"underline {SECONDARY_LIGHT}",
            "markdown.link_url": f"{SECONDARY_DIM}",
            "markdown.text": TEXT_PRIMARY,
            "markdown.em": f"italic {TEXT_PRIMARY}",
            "markdown.strong": f"bold {TEXT_PRIMARY}",
            "markdown.item.bullet": SECONDARY_COLOR,
            "markdown.item.number": SECONDARY_COLOR,
        }
    )


def get_welcome_header() -> Group:
    """Generate a styled welcome header with ASCII art and metadata.

    Creates a gradient-colored ASCII art header for the Astro CLI application
    followed by metadata describing the version and helpful command hints.

    Returns:
        Group: Rich render group containing the header and metadata rows.
    """
    # Get information
    version_value = get_astro_version().split("+")[0]
    version = f"[{TEXT_SECONDARY}]v[/{TEXT_SECONDARY}][{PRIMARY_COLOR}]{version_value}[/{PRIMARY_COLOR}]"
    organization = (
        f"[{PRIMARY_COLOR}][italic]Radio Astronomy Techniques and Technologies[/italic][{PRIMARY_COLOR}] "
        f"[{TEXT_SECONDARY}]([/{TEXT_SECONDARY}]"
        f"[{PRIMARY_COLOR}]RATT[/{PRIMARY_COLOR}]"
        f"[{TEXT_SECONDARY}])[/{TEXT_SECONDARY}]"
    )
    help_cmd = f"[bold {SECONDARY_COLOR}]/help[/bold {SECONDARY_COLOR}]"
    quit_cmd = f"[bold {SECONDARY_COLOR}]/quit[/bold {SECONDARY_COLOR}]"

    # ASCII art header
    welcome_header = _render_banner_with_starfield(
        textwrap.dedent("""

         █████╗ ███████╗████████╗██████╗  ██████╗      ██████╗██╗     ██╗
        ██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗    ██╔════╝██║     ██║
        ███████║███████╗   ██║   ██████╔╝██║   ██║    ██║     ██║     ██║
        ██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║    ██║     ██║     ██║
        ██║  ██║███████║   ██║   ██║  ██║╚██████╔╝    ╚██████╗███████╗██║
        ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝      ╚═════╝╚══════╝╚═╝

        """)
    )

    # Apply gradient to ASCII art
    header_lines = welcome_header.split("\n")
    gradient_colors = _create_gradient(
        SECONDARY_COLOR, PRIMARY_COLOR, len(header_lines)
    )
    colored_header = "\n".join(
        f"[{color}]{line}[/{color}]"
        for line, color in zip(header_lines, gradient_colors)
    )

    # Colorify
    _pick_star_color = {
        ".": PRIMARY_LIGHT,
        "·": PRIMARY_COLOR,
        "˙": SECONDARY_DARK,
        "•": SECONDARY_LIGHT,
    }
    for ch, color in _pick_star_color.items():
        colored_header = colored_header.replace(ch, f"[{color}]{ch}[/{color}]")

    angle_chars = ["╗", "╔", "╝", "╚", "║", "═"]
    for char in angle_chars:
        colored_header = colored_header.replace(
            char, f"[{PRIMARY_DARK}]{char}[/{PRIMARY_DARK}]"
        )

    # Build metadata section
    sub_header_1 = RichAlign.center(
        f"[{TEXT_SECONDARY}]by[/{TEXT_SECONDARY}] {organization}[{TEXT_SECONDARY}],[/{TEXT_SECONDARY}] {version}"
    )
    sub_header_2 = RichAlign.center(
        f"[{TEXT_SECONDARY}]Type [/{TEXT_SECONDARY}]{help_cmd}[{TEXT_SECONDARY}] for available commands or [/{TEXT_SECONDARY}]{quit_cmd}[{TEXT_SECONDARY}] to exit[/{TEXT_SECONDARY}]"
    )

    # Return as centred
    return Group(colored_header, sub_header_1, sub_header_2)


# --- Pygments → PTK style ---
_COMMAND_BASE = {
    "Name.Label": f"noreverse bg:default bold fg:{SECONDARY_COLOR}",
    "Name.Label.selected": f"noreverse bg:{BG_SECONDARY} bold fg:{SECONDARY_LIGHT}",
    "Name.Tag": f"noreverse bg:default bold underline fg:{PRIMARY_COLOR}",
    "Name.Tag.selected": f"noreverse bg:{BG_SECONDARY} bold fg:{PRIMARY_LIGHT}",
}

# Keep tokens minimal and stable for your RegexLexer.
_PYGMENTS_BASE = {
    Text: TEXT_PRIMARY,
    Name.Label: _COMMAND_BASE["Name.Label"],  # known slash commands
    Name.Tag: _COMMAND_BASE["Name.Tag"],  # known hashtags
    Generic.Error: f"fg:{ERROR_RED}",  # unknown commands/errors
}


# PTK class selectors for UI elements (completion menu, toolbar, etc.)
_PTK_UI = {
    # Input line and toolbar
    "prompt": f"bold {PRIMARY_COLOR}",
    "bottom-toolbar": f"noreverse {PRIMARY_COLOR}",
    # Base container (applies to both columns in COLUMN mode)
    "completion-menu": "bg:default",
    "completion-menu.completion": f"fg:{BG_PRIMARY}",
    "completion-menu.completion.current": f"fg:{BG_SECONDARY}",
    "completion-menu.meta": "bg:default",
    "completion-menu.meta.completion": f"fg:{TEXT_SECONDARY} bg:default",
    "completion-menu.meta.completion.current": f"bg:{BG_SECONDARY} fg:{TEXT_PRIMARY}",
    "completion-menu.completion fuzzymatch.outside": f"fg:{TEXT_SECONDARY}",
    # Search / validation prompts
    "search": INFO_BLUE,
    "validation-toolbar": f"bg:{BG_SECONDARY} fg:{ERROR_RED}",
    # Cursor
    "cursor": SECONDARY_LIGHT,
}


def make_prompt_style() -> BaseStyle:
    """Create the merged style used by the primary prompt session.

    Returns:
        BaseStyle: Prompt_toolkit style containing pygments tokens and UI
        selectors.
    """
    pyg_style = style_from_pygments_dict(_PYGMENTS_BASE)
    ptk_ui = Style.from_dict(_PTK_UI)
    return merge_styles([pyg_style, ptk_ui])


def make_slash_lexer(
    command_aliases: Iterable[str], hashtag_aliases: Iterable[str]
) -> type[RegexLexer]:
    """Create a RegexLexer that highlights known slash commands.

    Highlights valid slash commands at the start of input or after whitespace.
    All other text is rendered as plain text. Unknown slash commands are marked
    as errors.

    Args:
        command_aliases (Iterable[str]): Iterable of command strings for
            example "/help" or "/init".
        hashtag_aliases (Iterable[str]): Iterable of hashtag context options for
            example "#history".

    Returns:
        type[RegexLexer]: Lexer subclass that recognises known commands and
        hashtags.
    """
    command_set = set(command_aliases)
    hashtag_set = set(hashtag_aliases)

    command_alts = "|".join(sorted(re.escape(alias) for alias in command_set))
    hashtag_alts = "|".join(sorted(re.escape(alias) for alias in hashtag_set))

    command_re = rf"(?P<cmd>{command_alts})"
    hashtag_re = rf"(?P<tag>{hashtag_alts})"

    class ContextLexer(RegexLexer):
        """Lexer for slash commands and hashtags syntax highlighting."""

        name = "Context"
        tokens = {
            "root": [
                # Leading whitespace
                (r"^\s+", Text),
                # Known command at start (with optional leading whitespace)
                (r"^" + command_re + r"(?=\s|$)", Name.Label, "after_command"),
                # Unknown command at start
                (r"^/\S+", Generic.Error, "after_command"),
                # Known hashtag at start
                (r"^" + hashtag_re + r"(?=\s|$)", Name.Tag, "normal_text"),
                # Unknown hashtag at start
                (r"^#\S+", Generic.Error, "normal_text"),
                # Anything else at start is normal text
                (r"\S+", Text, "normal_text"),
            ],
            "after_command": [
                # After a slash command, everything is an error (no text allowed)
                (r"\s+", Text),
                (r"/\S*", Generic.Error),
                (r"#\S*", Generic.Error),
                (r"\S+", Generic.Error),
            ],
            "normal_text": [
                # Known hashtag
                (hashtag_re + r"(?=\s|$)", Name.Tag),
                # Unknown hashtag
                (r"#\S+", Generic.Error),
                # Whitespace
                (r"\s+", Text),
                # Any slash in the middle of normal text is an error
                (r"/\S*", Generic.Error),
                # Continue with normal text
                (r"\S+", Text),
            ],
        }

    return ContextLexer


class StyledCompleter(Completer):
    """Wrapper completer that applies styling to completion items.

    Wraps another completer and applies consistent styling to match the inline
    syntax highlighting. Used to ensure completion menu items use the same
    colors as their inline counterparts.
    """

    def __init__(
        self, base_completer: Completer, style: str = "", selected_style: str = ""
    ):
        """Initialize the styled completer.

        Args:
            base_completer (Completer): Underlying completer to wrap.
            style (str): Style string applied to completion text. Defaults to
                an empty string.
            selected_style (str): Style applied to the currently selected
                completion entry. Defaults to an empty string.
        """
        self._base_completer = base_completer
        self._style = style
        self._selected_style = selected_style

    def get_completions(self, document, complete_event):
        """Yield completions with applied styling.

        Args:
            document (Document): Prompt_toolkit document describing the input
                buffer.
            complete_event (CompleteEvent): Completion trigger event.

        Yields:
            Completion: Completion objects with styled display text.
        """

        for completion in self._base_completer.get_completions(
            document, complete_event
        ):
            yield Completion(
                text=completion.text,
                start_position=completion.start_position,
                style=self._style,
                display_meta=completion.display_meta,
                selected_style=self._selected_style,
            )


def make_slash_command_completer(
    commands: StrDict,
) -> StyledCompleter:
    """Create a styled completer for slash commands.

    Args:
        commands (StrDict): Mapping of command strings to descriptions.

    Returns:
        StyledCompleter: Styled completer configured for slash commands.
    """
    base_completer = FuzzyWordCompleter(list(commands.keys()), meta_dict=commands)
    return StyledCompleter(
        base_completer=base_completer,
        style=_COMMAND_BASE["Name.Label"],
        selected_style=_COMMAND_BASE["Name.Label.selected"],
    )


def make_hashtag_completer(
    hashtags: StrDict,
) -> StyledCompleter:
    """Create a styled completer for hashtags.

    Args:
        hashtags (StrDict): Mapping of hashtag strings to descriptions.

    Returns:
        StyledCompleter: Styled completer configured for hashtags.
    """
    base_completer = FuzzyWordCompleter(list(hashtags.keys()), meta_dict=hashtags)
    return StyledCompleter(
        base_completer=base_completer,
        style=_COMMAND_BASE["Name.Tag"],
        selected_style=_COMMAND_BASE["Name.Tag.selected"],
    )


class MultiCompleter(Completer):
    """Completer that delegates to different completers based on input prefix."""

    def __init__(
        self,
        slash_completer: Completer,
        hashtag_completer: Completer,
    ):
        """Initialize the multi-completer.

        Args:
            slash_completer (Completer): Completer handling slash commands.
            hashtag_completer (Completer): Completer handling hashtags.
        """
        self._slash_completer = slash_completer
        self._hashtag_completer = hashtag_completer

    def get_completions(self, document, complete_event):
        """Get completions based on the current input context.

        Args:
            document (Document): Prompt_toolkit document providing the text
                context.
            complete_event (CompleteEvent): Completion trigger event.

        Yields:
            Completion: Completion objects from the appropriate completer.
        """
        text = document.text_before_cursor

        # Empty input - show slash commands
        if not text or text.isspace():
            yield from self._slash_completer.get_completions(document, complete_event)
            return

        # Get the word currently being typed (text from last space to cursor)
        last_space_idx = text.rfind(" ")
        current_word = text[last_space_idx + 1 :] if last_space_idx != -1 else text

        # No current word being typed (cursor right after space) - no completions
        if not current_word:
            return

        # Determine which completer to use based on the first character of current word
        first_char = current_word[0]

        if first_char == "/":
            # Only show slash completions at the very start (no spaces before)
            if last_space_idx == -1:
                yield from self._slash_completer.get_completions(
                    document, complete_event
                )
        elif first_char == "#":
            # Show hashtag completions anywhere except after a slash command
            if not text.lstrip().startswith("/"):
                yield from self._hashtag_completer.get_completions(
                    document, complete_event
                )


class _CommandInputValidator(Validator):
    """Validate slash commands and hashtags against known registries."""

    def __init__(self, known_commands: set[str], known_hashtags: set[str]):
        """Initialise the validator with known command and hashtag registries.

        Args:
            known_commands (set[str]): Registered command tokens.
            known_hashtags (set[str]): Registered hashtag tokens.
        """

        self._commands = known_commands
        self._hashtags = known_hashtags

    def validate(self, document) -> None:  # noqa: D401 - required signature
        """Validate the current input document.

        Args:
            document (Document): Prompt_toolkit document under validation.

        Raises:
            ValidationError: Raised when an unknown command, hashtag, or
                unexpected argument is encountered.
        """

        text = document.text
        stripped = text.strip()
        if not stripped:
            return

        words = stripped.split()
        if stripped.lstrip().startswith("/"):
            command_token = words[0]
            if command_token not in self._commands:
                raise ValidationError(
                    message=f"Unknown command {command_token!r}. Use /help for options.",
                    cursor_position=len(command_token),
                )
            if len(words) > 1:
                raise ValidationError(
                    message=f"Command {command_token!r} does not accept arguments.",
                    cursor_position=len(document.text),
                )

        for token in words:
            if token.startswith("#") and token not in self._hashtags:
                raise ValidationError(
                    message=f"Unknown hashtag {token!r}.",
                    cursor_position=text.find(token) + len(token),
                )


_PROVIDER_COLOR_MAP = {
    "openai": ("#fcfdfc", "#0fa37e"),
    "anthropic": ("#fdfdf6", "#d87657"),
    "ollama": ("#fffeff", "#fffeff"),
}


def _format_model(model: ModelDetails) -> HTML:
    """Format model details into an HTML snippet for completion menus.

    Args:
        model (ModelDetails): Model metadata to format.

    Returns:
        HTML: Styled HTML snippet containing provider, model, and variant.

    Raises:
        ValueError: Raised when the provider is not registered in the provider
            color map.
    """

    if model.provider not in _PROVIDER_COLOR_MAP:
        raise _loggy.ValueError(f"Model provider {model.provider!r} not in color map")
    primary, secondary = _PROVIDER_COLOR_MAP[model.provider]
    provider_str = f'<style fg="{secondary}">{model.provider}</style>'
    model_str = f'<style fg="{primary}">{model.name}</style>'
    separator_str = f'<style fg="{TEXT_SECONDARY}">:</style>'

    if model.variant is None:
        return HTML(provider_str + separator_str + model_str)
    else:
        variant_str = f'<style fg="{SECONDARY_LIGHT}">{model.variant}</style>'
        return HTML(
            provider_str + separator_str + model_str + separator_str + variant_str
        )


def create_model_options() -> list[tuple[str, HTML]]:
    """Create formatted model options for the model picker.

    Returns:
        list[tuple[str, HTML]]: Sequence mapping model identifiers to formatted
        HTML representations.
    """

    models = KnownModels._identifier_to_details.values()
    return [(model.to_identifier(), _format_model(model)) for model in models]


def html_obj_to_rich_format(html_obj: HTML) -> str:
    """Convert a prompt_toolkit HTML object into Rich markup.

    Args:
        html_obj (HTML): Prompt_toolkit HTML object to convert.

    Returns:
        str: Equivalent Rich markup string.
    """

    def parse_style(style_str: str) -> tuple[str | None, str | None]:
        """
        style_str ~ "class:x,y fg:#ffffff bg:#000000"
        return (fg, bg) as strings without extra processing.
        """
        fg = None
        bg = None
        for token in style_str.split():
            if token.startswith("fg:"):
                fg = token[3:]
            elif token.startswith("bg:"):
                bg = token[3:]
        return fg, bg

    def escape_rich(text: str) -> str:
        """
        Escape '[' and ']' for Rich markup.
        Rich requires '[' -> '\\[' and ']' -> '\\]'.
        """
        return text.replace("[", r"\[").replace("]", r"\]")

    # Step 1. Get (style, text) spans from prompt_toolkit
    spans = html_obj.__pt_formatted_text__()

    # Step 2. Convert to list[(fg,bg,text)]
    colored_spans: list[tuple[str | None, str | None, str]] = []
    for span in spans:
        # OneStyleAndTextTuple can be (style, text) or (style, text, mouse_handler)
        style_str = span[0]
        text = span[1]
        fg, bg = parse_style(style_str or "")
        colored_spans.append((fg, bg, text))

    # Step 3. Merge consecutive spans with same (fg,bg)
    merged: list[tuple[str | None, str | None, str]] = []
    for fg, bg, text in colored_spans:
        if merged and merged[-1][0] == fg and merged[-1][1] == bg:
            old_fg, old_bg, old_text = merged[-1]
            merged[-1] = (old_fg, old_bg, old_text + text)
        else:
            merged.append((fg, bg, text))

    # Step 4. Build Rich string
    out_parts: list[str] = []
    for fg, bg, text in merged:
        body = escape_rich(text)
        if fg and bg:
            tag = f"{fg} on {bg}"
            out_parts.append(f"[{tag}]{body}[/{tag}]")
        elif fg:
            out_parts.append(f"[{fg}]{body}[/{fg}]")
        elif bg:
            # Rich supports "on color" as a style too. Keep fg blank.
            tag = f"on {bg}"
            out_parts.append(f"[{tag}]{body}[/{tag}]")
        else:
            out_parts.append(body)

    return "".join(out_parts)


_PTK_MODEL_SEARCH_UI = {
    # Input line and toolbar
    "prompt": f"bold {TEXT_SECONDARY}",
    # Base container (applies to both columns in COLUMN mode)
    "completion-menu": "bg:default",
    "completion-menu.completion": f"fg:{BG_PRIMARY}",
    "completion-menu.completion.current": f"fg:{BG_SECONDARY}",
    "completion-menu.meta": "bg:default",
    "completion-menu.meta.completion": f"fg:{TEXT_SECONDARY} bg:default",
    "completion-menu.meta.completion.current": f"bg:{BG_SECONDARY} fg:{TEXT_PRIMARY}",
    "completion-menu.completion fuzzymatch.inside": f"fg:{TEXT_PRIMARY}",
    "completion-menu.completion fuzzymatch.outside": f"fg:{TEXT_SECONDARY}",
    # Cursor
    "cursor": SECONDARY_LIGHT,
}


def make_model_search_style() -> BaseStyle:
    """Create the style used for the interactive model search prompt.

    Returns:
        BaseStyle: Prompt_toolkit style applied during model selection.
    """

    return Style.from_dict(_PTK_MODEL_SEARCH_UI)


def escape_rich_markup(text: str) -> str:
    """Escape special characters in Rich markup.

    Args:
        text (str): Input text containing potential Rich markup.

    Returns:
        str: Text with special characters escaped for Rich.
    """
    return text.replace("[", r"\[").replace("]", r"\]")


if __name__ == "__main__":
    text = "Hello, World!"
    prompt(_apply_ptk_style1(text, fg=PRIMARY_COLOR))
    prompt(_apply_ptk_style1(text, bg=PRIMARY_LIGHT))
    prompt(_apply_ptk_style1(text, bold=True))
    prompt(_apply_ptk_style1(text, italic=True))
    prompt(_apply_ptk_style1(text, decoration="underline"))
    prompt(_apply_ptk_style1(text, decoration="strikethrough"))
    prompt(
        _apply_ptk_style1(
            text,
            fg=PRIMARY_COLOR,
            bg=BG_SECONDARY,
            bold=True,
            italic=True,
            decoration="underline",
        )
    )

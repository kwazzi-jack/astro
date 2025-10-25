# --- Internal Imports ---
import bisect
import itertools
import math
import random
import re
import shutil
from collections.abc import Callable, Iterable

# --- External Imports ---
from colour import Color
from prompt_toolkit import HTML
from prompt_toolkit.application.current import get_app
from prompt_toolkit.completion import Completer, Completion, FuzzyWordCompleter
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import BaseStyle, Style, merge_styles
from prompt_toolkit.styles.pygments import style_from_pygments_dict
from pygments.lexer import RegexLexer
from pygments.token import Generic, Name, Text
from rich.align import Align as RichAlign
from rich.console import Group

# --- Local Imports ---
from astro.llms.base import KnownModels, ModelDetails
from astro.loggings.base import get_loggy
from astro.meta import get_astro_version
from astro.utilities.display import get_terminal_width

# --- GLOBALS ---
_loggy = get_loggy(__file__)


def _centre_line_to_terminal(line: str, width: int) -> str:
    line_length = len(line)
    padding = (width - line_length) // 2
    print(f"{padding=}")
    return " " * padding + line


def _centre_text_to_terminal(text: str) -> str:
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
        start_hex: Starting color in hex format (e.g., "#8b5cf6")
        end_hex: Ending color in hex format (e.g., "#a78bfa")
        steps: Number of steps in the gradient

    Returns:
        List of hex color strings representing the gradient
    """
    start = Color(start_hex)
    colors = list(start.range_to(Color(end_hex), steps))
    return [color.hex_l for color in colors]


def _make_star_sampler(weights: dict[str, int]) -> Callable[[random.Random], str]:
    """Return a weighted sampler over width-1 star glyphs."""
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
    """Sample from Poisson(lam) with Knuth for small λ, normal approx for large λ."""
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
    """Generate a clustered 2D starfield as a grid of glyphs."""
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
    lines = text.splitlines()
    max_length = max(len(line) for line in lines) if lines else 0
    squarified_lines = [line.center(max_length) for line in lines]
    return "\n".join(squarified_lines)


def _overlay_banner_on_starfield(
    banner: str,
    star_grid: list[list[str]],
) -> str:
    """Overlay the banner into the center of star_grid. Only replaces cells where banner has non-space."""
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
    """High-level API: build a 2D clustered starfield and overlay banner centered within it."""
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


def get_welcome_header() -> Group:
    """Generate a styled welcome header with ASCII art and metadata.

    Creates a gradient-colored ASCII art header for the ASTRO CLI application,
    followed by version information, organization details, and helpful command hints.

    Args:
        version: Application version string (e.g., "1.0.0")
        organization: Organization or author name

    Returns:
        Formatted welcome header string with Rich markup for colored output
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
        """

 █████╗ ███████╗████████╗██████╗  ██████╗      ██████╗██╗     ██╗
██╔══██╗██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗    ██╔════╝██║     ██║
███████║███████╗   ██║   ██████╔╝██║   ██║    ██║     ██║     ██║
██╔══██║╚════██║   ██║   ██╔══██╗██║   ██║    ██║     ██║     ██║
██║  ██║███████║   ██║   ██║  ██║╚██████╔╝    ╚██████╗███████╗██║
╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝      ╚═════╝╚══════╝╚═╝

"""
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
    """Return a merged Style for PromptSession(style=...).


    - Pygments tokens from your lexer
    - PTK UI classes for menus/toolbars
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
        command_aliases (Iterable[str]): Iterable of command strings (e.g. /help, /init).
        hashtag_aliases (Iterable[str]): Iterable of hashtag context options (e.g. #base.py, #MyClass)

    Returns:
        type[RegexLexer]: A RegexLexer subclass that recognizes known commands and hashtags.
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
            base_completer: The underlying completer to wrap.
            style: Style string to apply to completion text (e.g., "bold fg:#6c97d0").
            meta_style: Optional style string for metadata text.
        """
        self._base_completer = base_completer
        self._style = style
        self._selected_style = selected_style
        self._app = get_app()

    def get_completions(self, document, complete_event):
        """Get completions with applied styling.

        Args:
            document: The current document.
            complete_event: The completion event.

        Yields:
            Completion objects with styled display text.
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
    commands: dict[str, str],
) -> StyledCompleter:
    """Create a styled completer for slash commands.

    Args:
        commands: Dictionary mapping command strings to their descriptions.

    Returns:
        A StyledCompleter with slash command styling.
    """
    base_completer = FuzzyWordCompleter(list(commands.keys()), meta_dict=commands)
    return StyledCompleter(
        base_completer=base_completer,
        style=_COMMAND_BASE["Name.Label"],
        selected_style=_COMMAND_BASE["Name.Label.selected"],
    )


def make_hashtag_completer(
    hashtags: dict[str, str],
) -> StyledCompleter:
    """Create a styled completer for hashtags.

    Args:
        hashtags: Dictionary mapping hashtag strings to their descriptions.

    Returns:
        A StyledCompleter with hashtag styling.
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
            slash_completer: Completer for slash commands.
            hashtag_completer: Completer for hashtags.
        """
        self._slash_completer = slash_completer
        self._hashtag_completer = hashtag_completer

    def get_completions(self, document, complete_event):
        """Get completions based on the current input context.

        Args:
            document: The current document.
            complete_event: The completion event.

        Yields:
            Completion objects from the appropriate completer.
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


_PROVIDER_COLOR_MAP = {
    "openai": ("#fcfdfc", "#0fa37e"),
    "anthropic": ("#fdfdf6", "#d87657"),
    "ollama": ("#fffeff", "#fffeff"),
}


def _format_model(model: ModelDetails) -> HTML:
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
    models = KnownModels._identifier_to_details.values()
    return [(model.to_identifier(), _format_model(model)) for model in models]


def html_obj_to_rich_format(html_obj: HTML) -> str:
    """
    Convert a prompt_toolkit.HTML object into a Rich markup string.

    Rules:
    - Keep only explicit fg:/bg: colors.
    - Ignore class:... and other style tokens.
    - Merge adjacent spans that resolve to the same (fg, bg).
    - Map:
        fg only        -> "[fg]text[/fg]"
        fg and bg      -> "[fg on bg]text[/fg on bg]"
        neither        -> "text"
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
    return Style.from_dict(_PTK_MODEL_SEARCH_UI)

"""Display utilities supporting the Astro CLI."""

# --- Internal Imports ---
from shutil import get_terminal_size

# --- Globals ---
MIN_TERMINAL_WIDTH = 40


def get_terminal_width() -> int:
    """Return the available terminal width with a guarded minimum.

    Returns:
        int: Terminal width in characters; always at least 40 to maintain layout.

    Examples:
        >>> width = get_terminal_width()
        >>> width >= MIN_TERMINAL_WIDTH
        True
    """
    terminal_width = get_terminal_size().columns
    return (
        terminal_width if terminal_width >= MIN_TERMINAL_WIDTH else MIN_TERMINAL_WIDTH
    )

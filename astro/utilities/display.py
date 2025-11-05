"""Display utilities supporting the Astro CLI."""

# --- Internal Imports ---
from shutil import get_terminal_size

# --- Local Imports ---
from astro.logger import get_loggy

# --- Globals ---
_loggy = get_loggy(__file__)
MIN_TERMINAL_WIDTH = 40
_loggy.debug("Minimum terminal width set", MIN_TERMINAL_WIDTH=MIN_TERMINAL_WIDTH)


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

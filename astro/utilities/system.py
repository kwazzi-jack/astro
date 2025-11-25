"""System and environment detection utilities for Astro."""

# --- Internal Imports ---
import os

# -- System Helper Functions ---


def get_users_name() -> str:
    """Return the login name for the current user.

    Returns:
        str: Username reported by the operating system.

    Examples:
        >>> get_users_name()  # doctest: +SKIP
        'astro-user'
    """
    return os.getlogin()

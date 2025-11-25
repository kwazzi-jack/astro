# -- Internal Imports ---
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

# --- External Imports ---
from pydantic_ai import UserContent

# --- Output Type Aliases ---
StreamIn: TypeAlias = str | Sequence[UserContent] | None

__all__ = [
    "StreamIn",
]

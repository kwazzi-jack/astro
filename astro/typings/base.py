"""Core type aliases and helper utilities for Astro."""

# --- Internal Imports ---
from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args

# --- External Imports ---
from prompt_toolkit import HTML
from pydantic import TypeAdapter
from pydantic_ai import AgentRunResultEvent, AgentStreamEvent, ModelMessage

# --- Local Imports ---
if TYPE_CHECKING:
    from astro.typings.callables import (
        AnyFn,
        DateTimeFn,
        FloatFn,
        HTMLFn,
        InlineFn,
        MarkdownFn,
        MarkupFn,
        StreamFn,
        StrFn,
    )
    from astro.typings.inputs import StreamIn
    from astro.typings.outputs import AgentOutputType, StreamOut

# --- Generic Types & Type Aliases ---
StrPath: TypeAlias = str | Path
PTKDecoration: TypeAlias = Literal["underline", "strikethrough", "none"]
PydanticEvent = AgentStreamEvent | AgentRunResultEvent[Any]
PathKind = Literal[
    "file",
    "directory",
    "symlink",
    "socket",
    "fifo",
    "block_device",
    "character_device",
    "missing",
    "unknown",
]

# Dictionaries / Structs
StrDict: TypeAlias = dict[str, str]
PathDict: TypeAlias = dict[str, Path]
NamedDict: TypeAlias = dict[str, Any]
HTMLDict: TypeAlias = dict[str, HTML]

# Lists / Tuples
MessageList: TypeAlias = list[ModelMessage]

# Special
ArgsAdapter = TypeAdapter(NamedDict)


# --- General & Type Helper Functions ---
def literal_to_list(literal_type: Any) -> list[Any]:
    """Convert a Literal type alias into its list of contained values.

    Args:
        literal_type (Any): Literal annotation to extract values from.

    Returns:
        list[Any]: All values defined inside the literal annotation.
    """

    return [value for value in get_args(literal_type)]


def secretify(value: Any) -> str:
    """Obscure the middle portion of a value converted to string.

    Args:
        value (Any): Value to anonymize.

    Returns:
        str: Obscured representation that keeps the prefix and suffix visible.
    """

    value_str = str(value)
    length = len(value_str)

    if length <= 8:
        return "*" * 8

    if length > 15:
        prefix_len = 4
        suffix_len = 4
    else:
        uncovered = length - 8
        prefix_len = uncovered // 2
        suffix_len = uncovered - prefix_len

    prefix = value_str[:prefix_len]
    suffix = value_str[length - suffix_len :]
    num_asterisks = length - prefix_len - suffix_len

    return f"{prefix}{'*' * num_asterisks}{suffix}"


def str_dict_to_path_dict(contents: StrDict) -> PathDict:
    """Convert a dictionary of string paths into resolved Path objects."""

    try:
        return {key: Path(value).resolve() for key, value in contents.items()}
    except Exception as error:  # pragma: no cover - defensive
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def path_dict_to_str_dict(contents: PathDict) -> StrDict:
    """Convert a dictionary of Path objects into string representations."""

    try:
        return {key: str(value) for key, value in contents.items()}
    except Exception as error:  # pragma: no cover - defensive
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def type_name(obj: Any) -> str:
    """Return the class name for an object or type."""

    if isinstance(obj, type):
        return obj.__name__
    return type(obj).__name__


def options_to_str(values: Sequence[str], with_repr: bool = False) -> str:
    """Convert a sequence of strings into a readable options list."""

    if len(values) == 0:
        return "''" if with_repr else ""
    if len(values) == 1:
        return repr(values[0]) if with_repr else values[0]
    if with_repr:
        return (
            ", ".join(repr(value) for value in values[:-1]) + " or " + repr(values[-1])
        )
    return ", ".join(values[:-1]) + " or " + values[-1]


def type_options(objects: Sequence[Any]) -> list[str]:
    """Return a list of class names for the provided sequence."""

    return [type_name(obj) for obj in objects]


def get_path_type(path: Path) -> PathKind:
    """Determine the filesystem entry type for a path."""

    if not path.exists():
        return "missing"
    if path.is_file():
        return "file"
    if path.is_dir():
        return "directory"
    if path.is_symlink():
        return "symlink"
    if path.is_socket():
        return "socket"
    if path.is_fifo():
        return "fifo"
    if path.is_block_device():
        return "block_device"
    if path.is_char_device():
        return "character_device"
    return "unknown"


# --- JSON Encoders ---
def _datetime_json_encoder(dt: datetime) -> str:
    """Serialize a datetime instance into an ISO 8601 string."""

    return dt.isoformat()


def _path_json_encoder(path: Path) -> str:
    """Serialize a Path instance into its string representation."""

    return str(path)


__all__ = [
    "ArgsAdapter",
    "AgentOutputType",
    "AnyFn",
    "StreamFn",
    "StreamIn",
    "StreamOut",
    "DateTimeFn",
    "FloatFn",
    "HTMLDict",
    "HTMLFn",
    "InlineFn",
    "MarkdownFn",
    "MarkupFn",
    "MessageList",
    "NamedDict",
    "PTKDecoration",
    "PathDict",
    "PathKind",
    "PydanticEvent",
    "StrDict",
    "StrPath",
    "StrFn",
    "get_path_type",
    "literal_to_list",
    "options_to_str",
    "path_dict_to_str_dict",
    "secretify",
    "str_dict_to_path_dict",
    "type_name",
    "type_options",
]

# --- Internal Imports ---
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypeAliasType, get_args

# --- External Imports ---
from prompt_toolkit import HTML
from pydantic import BaseModel
from pydantic_ai import AgentRunResultEvent, AgentStreamEvent, ModelMessage

# --- Local Imports ---
# ...

# --- Generic Types & Type Aliases ---
StrPath: TypeAlias = str | Path
PTKDecoration: TypeAlias = Literal["underline", "strikethrough", "none"]

# Dictionaries / Structs
StrDict: TypeAlias = dict[str, str]
PathDict: TypeAlias = dict[str, Path]
NamedDict: TypeAlias = dict[str, Any]
HTMLDict: TypeAlias = dict[str, HTML]
SchemaLike: TypeAlias = type[BaseModel] | BaseModel | dict[str, Any]

# Lists / Tuples
MessageList: TypeAlias = list[ModelMessage]

# Functions
AnyFactory: TypeAlias = Callable[[], Any]
AsyncChatFunction: TypeAlias = Callable[
    [str], AsyncIterator[AgentStreamEvent | AgentRunResultEvent[str]]
]
DateTimeFactory: TypeAlias = Callable[[], datetime]
FloatFactory: TypeAlias = Callable[[], float]
HTMLFactory: TypeAlias = Callable[[], HTML]
InlineFn: TypeAlias = Callable[[], None]
StringFactory: TypeAlias = Callable[[], str]

# --- General & Type Helper Functions ---


def literal_to_list(literal_type: TypeAliasType) -> list[Any]:
    """Convert a Literal type alias into its list of contained values.

    Args:
        literal_type (TypeAliasType): Type alias created from a typing.Literal
            declaration.

    Returns:
        list[Any]: Values extracted from the literal alias.
    """

    return [value for value in get_args(literal_type)]


def secretify(value: Any) -> str:
    """Obscure the middle portion of a value converted to string.

    Args:
        value (Any): Value whose string representation should be masked.

    Returns:
        str: Obfuscated string that preserves only the leading and trailing
        characters.
    """
    value_str = str(value)
    length = len(value_str)

    if length <= 8:
        return "*" * 8

    if length > 15:
        prefix_len = 4
        suffix_len = 4
    else:
        # This logic covers lengths from 9 to 15, matching the original if-chain
        uncovered = length - 8
        prefix_len = uncovered // 2
        suffix_len = uncovered - prefix_len

    prefix = value_str[:prefix_len]
    suffix = value_str[length - suffix_len :]
    num_asterisks = length - prefix_len - suffix_len

    return f"{prefix}{'*' * num_asterisks}{suffix}"


def str_dict_to_path_dict(contents: StrDict) -> PathDict:
    """Convert a dictionary of string paths into resolved Path objects.

    Args:
        contents (StrDict): Mapping of identifiers to filesystem paths stored
            as strings.

    Returns:
        PathDict: Mapping of identifiers to resolved Path instances.

    Raises:
        ValueError: Raised when any provided path cannot be converted into a
            valid Path instance.
    """

    try:
        return {key: Path(value).resolve() for key, value in contents.items()}
    except Exception as error:
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def path_dict_to_str_dict(contents: PathDict) -> StrDict:
    """Convert a dictionary of Path objects into string representations.

    Args:
        contents (PathDict): Mapping of identifiers to Path instances.

    Returns:
        StrDict: Mapping of identifiers to string paths.

    Raises:
        ValueError: Raised when a path cannot be converted to a string.
    """

    try:
        return {key: str(value) for key, value in contents.items()}
    except Exception as error:
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def type_name(obj: Any) -> str:
    """Return the class name for an object or type.

    Args:
        obj (Any): Object or type instance to inspect.

    Returns:
        str: Deduced class name.
    """

    if isinstance(obj, type):
        return obj.__name__
    else:
        return type(obj).__name__


def options_to_str(values: Sequence[str], with_repr: bool = False) -> str:
    """Convert a sequence of strings into a readable options list.

    Args:
        values (Sequence[str]): Sequence of option strings to format.
        with_repr (bool): Optional flag indicating whether to wrap entries in
            repr formatting. Defaults to False.

    Returns:
        str: Formatted options string suitable for messaging.
    """

    if len(values) == 0:
        return "''" if with_repr else ""
    elif len(values) == 1:
        return repr(values[0]) if with_repr else values[0]
    else:
        if with_repr:
            return (
                ", ".join(repr(value) for value in values[:-1])
                + " or "
                + repr(values[-1])
            )
        else:
            return ", ".join(values[:-1]) + " or " + values[-1]


def type_options(objects: Sequence[Any]) -> list[str]:
    """Return a list of class names for the provided sequence.

    Args:
        objects (Sequence[Any]): Objects whose class names should be returned.

    Returns:
        list[str]: Class names associated with the provided objects.
    """

    return list(map(type_name, objects))


class PathKind(StrEnum):
    """Enumeration describing supported filesystem path classifications."""

    FILE = "file"
    DIRECTORY = "directory"
    SYMLINK = "symlink"
    SOCKET = "socket"
    FIFO = "fifo"
    BLOCK_DEVICE = "block_device"
    CHARACTER_DEVICE = "character_device"
    MISSING = "missing"
    UNKNOWN = "unknown"


def get_path_type(path: Path) -> PathKind:
    """Determine the filesystem entry type for a path.

    Args:
        path (Path): Filesystem path to inspect.

    Returns:
        PathKind: Enumerated path classification.
    """

    if not path.exists():
        return PathKind.MISSING
    if path.is_file():
        return PathKind.FILE
    if path.is_dir():
        return PathKind.DIRECTORY
    if path.is_symlink():
        return PathKind.SYMLINK
    if path.is_socket():
        return PathKind.SOCKET
    if path.is_fifo():
        return PathKind.FIFO
    if path.is_block_device():
        return PathKind.BLOCK_DEVICE
    if path.is_char_device():
        return PathKind.CHARACTER_DEVICE
    return PathKind.UNKNOWN


def _datetime_json_encoder(dt: datetime) -> str:
    """Serialize a datetime instance into an ISO 8601 string.

    Args:
        dt (datetime): Datetime value to serialize.

    Returns:
        str: ISO formatted datetime string.
    """

    return dt.isoformat()


def _path_json_encoder(path: Path) -> str:
    """Serialize a Path instance into its string representation.

    Args:
        path (Path): Path instance to serialize.

    Returns:
        str: String form of the provided path.
    """

    return str(path)


def _object_log_formatter(obj: Any) -> str:
    if isinstance(obj, str):
        return repr(obj)
    if isinstance(obj, datetime):
        return repr(_datetime_json_encoder(obj))
    if isinstance(obj, Path):
        return repr(_path_json_encoder(obj))
    return repr(obj)

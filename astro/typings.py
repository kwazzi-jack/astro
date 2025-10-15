# --- Internal Imports ---
import base64
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeAlias, TypeAliasType, TypeVar, get_args

# --- External Imports ---
import blake3
from pydantic import BaseModel, ConfigDict

# --- Local Imports ---

# --- Generic Types & Type Aliases ---
StrPath: TypeAlias = str | Path
StrDict: TypeAlias = dict[str, str]
PathDict: TypeAlias = dict[str, Path]
NamedDict: TypeAlias = dict[str, Any]
SchemaLike: TypeAlias = type[BaseModel] | BaseModel | dict[str, Any]
RecordableType = TypeVar("RecordableType", bound="RecordableModel")


def literal_to_list(literal_type: TypeAliasType) -> list[Any]:
    return [value for value in get_args(literal_type)]


def secretify(value: Any) -> str:
    """Obscures the middle part of a string, showing only the ends."""
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
    try:
        return {key: Path(value).resolve() for key, value in contents.items()}
    except Exception as error:
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def path_dict_to_str_dict(contents: PathDict) -> StrDict:
    try:
        return {key: str(value) for key, value in contents.items()}
    except Exception as error:
        raise ValueError(
            "Error while converting conversation index dictionary values to paths"
        ) from error


def type_name(obj: Any) -> str:
    if isinstance(obj, type):
        return obj.__name__
    else:
        return type(obj).__name__


def options_to_str(values: Sequence[str], with_repr: bool = False) -> str:
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


def options_to_repr_str(values: Sequence[str]) -> str:
    return options_to_str(list(map(repr, values)))


def type_options(objects: Sequence[Any]) -> list[str]:
    return list(map(type_name, objects))


class PathKind(StrEnum):
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
    return dt.isoformat()


def _path_json_encoder(path: Path) -> str:
    return str(path)


class RecordableModel(BaseModel, frozen=True):
    """Base model for recordable objects with hashing and dictionary utilities.

    This class provides methods for generating hashes and converting to dictionaries,
    suitable for database storage or serialization.

    Attributes:
        None (inherits from BaseModel; placeholder for any custom attributes).
    """

    model_config = ConfigDict(
        frozen=True,
        # These ensure consistent serialization:
        use_enum_values=False,  # Don't convert enums to their values
        json_encoders={
            datetime: _datetime_json_encoder,  # Consistent datetime format
            Path: _path_json_encoder,  # Convert Path to string
        },
    )

    def _compute_stable_hash(self) -> bytes:
        """Compute a stable blake3 hash that's consistent across systems.

        Returns:
            bytes: The raw blake3 hash bytes (32 bytes).
        """

        # Get JSON-serializable representation
        model_dict = self.model_dump(
            mode="json",
            exclude_none=False,
            exclude_defaults=False,
        )

        # Create deterministic JSON string
        # Brian - If something is wrong with hashing, its probably here
        json_str = json.dumps(
            model_dict,
            sort_keys=True,  # IMPORTANT: for consistent key order
            separators=(",", ";"),  # No whitespace
            ensure_ascii=True,  # Avoid encoding variations
            default=str,  # Fallback conversion
        )

        # Generate black3 hash
        return blake3.blake3(json_str.encode("utf-8")).digest()

    # OVERRIDE: pydantic.BaseModel.__hash__
    def __hash__(self) -> int:
        """Return the hash value of the model based on its stable hash.

        Returns:
            int: The hash value computed from the stable hash.
        """
        return int.from_bytes(self._compute_stable_hash(), byteorder="big")

    # OVERRIDE: pydantic.BaseModel.__eq__
    def __eq__(self, other: Any) -> bool:
        """Override of BaseModel.__eq__ to compare models based on their stable hash.

        Two models are considered equal if they are of the same type and have the same hash value,
        ensuring consistent equality based on content rather than identity.

        Args:
            other (Any): The object to compare with this model instance.

        Returns:
            bool: True if the objects are equal, False otherwise.
        """

        # Not the same type
        if not isinstance(other, type(self)):
            return False

        # Compare hashes
        return hash(self) == hash(other)

    @property
    def uid(self) -> str:
        """Return unique identifier based on the model's content hash."""
        return self.to_hex()

    def to_hex(self) -> str:
        return hex(hash(self))[2:]

    def to_base64(self) -> str:
        """Generate a base64-encoded representation of the model's stable hash.

        Returns:
            str: Base64-encoded string of the model's blake3 hash (44 characters).
        """
        return base64.b64encode(self._compute_stable_hash()).decode("ascii")

    def secret_uid(self) -> str:
        return secretify(self.uid)


def count_pydantic_fields(model_class: type[BaseModel]) -> int:
    return len(model_class.model_fields)


def get_class_import_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__name__}"


def get_class_from_import_path(import_path: str) -> type:
    components = import_path.split(".")
    module_path = ".".join(components[:-1])
    class_name = components[-1]

    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls


if __name__ == "__main__":
    ...

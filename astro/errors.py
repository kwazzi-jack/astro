from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from sqlalchemy.exc import IntegrityError as DBIntegrityError

from astro.typings import (
    HashableObject,
    ImmutableRecord,
    NamedDict,
    RecordableModel,
    StrPath,
    _options_to_str,
    get_path_type,
    type_name,
    type_options,
)
from astro.utilities.display import inline_code_format, inline_list_format

AstroErrorType = TypeVar("AstroErrorType", bound="BaseError")
PythonErrorType = TypeVar("PythonErrorType", bound=Exception)


class BaseError(ABC, Exception):
    """Base error class"""

    def __init__(self, *, message: str, extra: NamedDict | None = None):
        self._message = message
        self._extra = extra
        super().__init__(message)

    @property
    def message(self) -> str:
        return self._message

    @property
    def extra(self) -> NamedDict | None:
        return self._extra

    def to_log(self) -> NamedDict:
        return {
            "msg": self.message,
            "extra": self.extra,
        }


class SetupError(BaseError):
    """Raised when an error occurs during setup."""

    def __init__(self, *, cause: str | None = None, extra: NamedDict | None = None):
        # Construct parent
        super().__init__(message=f"Setup error occurred: {cause}", extra=extra)


class RecordableIdentityError(BaseError):
    """When the identity of two models are not the same when they should be."""

    def __init__(
        self,
        *,
        record: HashableObject,
        other_record: HashableObject,
        extra: NamedDict | None = None,
    ) -> None:
        # Model and record name
        record_type = type_name(record)
        other_record_type = type_name(other_record)

        record_hash = (
            hash(record) if isinstance(record, RecordableModel) else record.record_hash
        )
        other_record_hash = (
            hash(other_record)
            if isinstance(other_record, RecordableModel)
            else other_record.record_hash
        )

        # Error details
        extra = (extra or {}) | {
            "record_type": type(record),
            "record_hash": record_hash,
            "other_record_type": type(other_record),
            "other_record_hash": other_record_hash,
        }

        # Error message
        message = (
            f"Hash mistmatch between model {record_type} ({record_hash}) "
            f"and record {other_record_type} ({other_record_hash})"
        )

        # Construct parent
        super().__init__(message=message, extra=extra)


class ExpectedVarType(BaseError):
    """Raised when a variable's type doesn't match the expected type(s).

    This error provides detailed information about type validation failures,
    including the variable name, actual type received, and expected types.
    """

    def __init__(
        self,
        *,
        var_name: str,
        got: type,
        expected: Sequence[type] | type,
        extra: NamedDict | None = None,
    ):
        # Normalize expected to always be a sequence
        if not isinstance(expected, Sequence):
            expected_types = [expected]
        else:
            expected_types = list(expected)

        # Expected and got type names
        expected_types = type_options(expected_types)
        got_type = type_name(got)

        # Error details
        extra = (extra or {}) | {
            "var_name": var_name,
            "got_type": got_type,
            "expected_types": expected_types,
        }

        # Error message
        expected_str = inline_list_format(expected_types)
        message = (
            f"Expected `{var_name}` to be {expected_str}. Got `{got_type}` instead"
        )

        # Construct parent
        super().__init__(message=message, extra=extra)


class ExpectedTypeError(BaseError):
    """Raised when a type doesn't match the expected type(s) without variable context.

    This is the base class for type validation errors that don't involve specific variables.
    """

    def __init__(
        self,
        *,
        got: type,
        expected: Sequence[type] | type,
        extra: NamedDict | None = None,
    ):
        # Normalize expected to always be a sequence
        if not isinstance(expected, Sequence):
            expected_types = [expected]
        else:
            expected_types = list(expected)

        # Expected and got type names
        expected_types = type_options(expected_types)
        got_type = type_name(got)

        # Error details
        extra = (extra or {}) | {"got_type": got_type, "expected_types": expected_types}

        # Error message
        expected_str = inline_list_format(expected_types)
        message = f"Expected type to be {expected_str}. Got `{got_type}` instead"

        # Construct parent
        super().__init__(message=message, extra=extra)


class KeyTypeError(ExpectedVarType):
    """Raised when a key's type doesn't match the expected type(s)."""

    def __init__(
        self,
        *,
        got: type,
        expected: Sequence[type] | type,
        extra: NamedDict | None = None,
    ):
        super().__init__(var_name="key", got=got, expected=expected, extra=extra)


class KeyStrError(KeyTypeError):
    """Raised when a key is expected to be a string but isn't."""

    def __init__(self, *, got: type, extra: NamedDict | None = None):
        super().__init__(got=got, expected=str, extra=extra)


class ValueTypeError(ExpectedVarType):
    """Raised when a value's type doesn't match the expected type(s)."""

    def __init__(
        self,
        *,
        got: type,
        expected: Sequence[type] | type,
        extra: NamedDict | None = None,
    ):
        super().__init__(var_name="value", got=got, expected=expected, extra=extra)


class NoEntryKeyError(BaseError):
    def __init__(
        self,
        *,
        key_value: Any,
        sources: Sequence[str | RecordableModel | ImmutableRecord]
        | str
        | RecordableModel
        | ImmutableRecord
        | None = None,
        extra: NamedDict | None = None,
    ):
        # Error details
        extra = (extra or {}) | {"key_value": key_value}

        # Normalize sources to a sequence for consistent processing
        sources_seq: Sequence[str | RecordableModel | ImmutableRecord] = []
        if sources is not None:
            if isinstance(sources, Sequence) and not isinstance(sources, str):
                sources_seq = sources
            else:
                sources_seq = [sources]
            extra["sources"] = sources_seq

            # Prepare formatted source descriptions for message
            formatted_sources = []
            object_hashes = []
            for src in sources_seq:
                if isinstance(src, (RecordableModel, ImmutableRecord)):
                    object_type = type_name(type(src))
                    object_hash = (
                        hash(src)
                        if isinstance(src, RecordableModel)
                        else src.record_hash
                    )
                    formatted_sources.append(f"{object_type} ({object_hash})")
                    object_hashes.append(object_hash)
                elif isinstance(src, str):
                    formatted_sources.append(inline_code_format(src))
                else:
                    formatted_sources.append(repr(src))
            if object_hashes:
                extra["object_hash"] = (
                    object_hashes if len(object_hashes) > 1 else object_hashes[0]
                )
        else:
            formatted_sources = []

        # Error message
        message = f"No entry for key `{key_value}`"
        if formatted_sources:
            message += " in sources " + _options_to_str(formatted_sources)

        # Construct parent
        super().__init__(message=message, extra=extra)


class LoadError(BaseError):
    """Raised when an error occurs when loading."""

    def __init__(
        self,
        *,
        path_or_uid: StrPath | int | None = None,
        obj_or_key: RecordableModel | ImmutableRecord | Any | None = None,
        load_from: str | None = None,
        extra: NamedDict | None = None,
    ):
        # Normalize to Path if parseable
        if isinstance(path_or_uid, str) and ("/" in path_or_uid or "\\" in path_or_uid):
            path_or_uid = Path(path_or_uid)

        # Error details
        extra = extra or {}

        # Error message
        message = "Error occurred while loading key or object"

        # Object to save
        extra["object_or_key_type"] = type(obj_or_key)
        message += f" {type_name(obj_or_key)}"

        # Object has hash
        if isinstance(obj_or_key, (RecordableModel, ImmutableRecord)):
            object_hash = (
                hash(obj_or_key)
                if isinstance(obj_or_key, RecordableModel)
                else obj_or_key.record_hash
            )
            extra["object_hash"] = object_hash
            message += f" ({object_hash})"

        message += " to"

        # Loading source
        if isinstance(load_from, str) and len(load_from.strip()) > 0:
            extra["load_from"] = load_from
            message += f" {load_from}"

        # Add path
        if isinstance(path_or_uid, Path):
            extra["path"] = path_or_uid
            extra["name"] = path_or_uid.name
            extra["path_exists"] = path_or_uid.exists()
            extra["path_type"] = get_path_type(path_or_uid)
            message += f" {inline_code_format(str(path_or_uid))}"

        # Add uid
        elif path_or_uid is not None:
            extra["uid"] = path_or_uid
            message += f" using uid {inline_code_format(str(path_or_uid))}"

        # Construct parent
        super().__init__(message=message, extra=extra)


class SaveError(BaseError):
    """Raised when an error occurs when saving."""

    def __init__(
        self,
        *,
        path_or_uid: StrPath | None = None,
        obj_to_save: RecordableModel | ImmutableRecord | Any | None = None,
        save_to: str | None = None,
        extra: NamedDict | None = None,
    ):
        # Normalize to Path if parseable
        if isinstance(path_or_uid, str) and ("/" in path_or_uid or "\\" in path_or_uid):
            path_or_uid = Path(path_or_uid)

        # Error details
        extra = extra or {}

        # Error message
        message = "Error occurred while saving"

        # Object to save
        extra["object_type"] = type(obj_to_save)
        message += f" {type_name(obj_to_save)}"

        # Object has hash
        if isinstance(obj_to_save, (RecordableModel, ImmutableRecord)):
            object_hash = (
                hash(obj_to_save)
                if isinstance(obj_to_save, RecordableModel)
                else obj_to_save.record_hash
            )
            extra["object_hash"] = object_hash
            message += f" ({object_hash})"

        message += " to"

        # Saving source
        if isinstance(save_to, str) and len(save_to.strip()) > 0:
            extra["save_to"] = save_to
            message += f" {save_to}"

        # Add path
        if path_or_uid is not None and isinstance(path_or_uid, Path):
            extra["path"] = path_or_uid
            extra["name"] = path_or_uid.name
            extra["path_exists"] = path_or_uid.exists()
            extra["path_type"] = get_path_type(path_or_uid)
            message += f" {inline_code_format(str(path_or_uid))}"

        # Add uid
        elif path_or_uid is not None:
            extra["uid"] = path_or_uid
            message += f" using uid {inline_code_format(str(path_or_uid))}"

        # Construct parent
        super().__init__(message=message, extra=extra)


class ModelStoreError(BaseError):
    """Raised when an error occurs with a store operation."""

    def __init__(
        self,
        *,
        operation: str,
        store_name: str | None = None,
        extra: NamedDict | None = None,
    ):
        # Error details
        extra = extra or {}
        extra["operation"] = operation
        if store_name is not None:
            extra["store_name"] = store_name

        # Error message
        message = f"Error occurred during store operation: {operation}"
        if store_name is not None:
            message += f" on store {store_name}"

        # Construct parent
        super().__init__(message=message, extra=extra)


if __name__ == "__main__":
    import json

    from astro.typings import ImmutableRecord, RecordableModel

    obj = "hey there"
    error = SaveError(
        path_or_uid="/home/brian/PhD/astro/.gitignore", save_to="file", obj_to_save=obj
    )
    print(json.dumps(error.extra, indent=2, default=str))
    print()
    raise error

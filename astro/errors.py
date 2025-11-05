# --- Internal Imports ---
from abc import ABC
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

# --- Local Imports ---
from astro.typings import (
    NamedDict,
    StrPath,
    get_path_type,
    options_to_str,
    secretify,
    type_name,
    type_options,
)

# --- Globals ---
AstroErrorType = TypeVar("AstroErrorType", bound="AstroError")
PythonErrorType = TypeVar("PythonErrorType", bound=Exception)


class AstroError(ABC, Exception):
    """Base error class"""

    def __init__(
        self,
        *,
        message: str,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        self._message = message
        self._extra = extra
        super().__init__(message)
        self.__cause__ = caused_by if caused_by is not None else None

    @property
    def message(self) -> str:
        return self._message

    @property
    def extra(self) -> NamedDict | None:
        return self._extra

    @property
    def caused_by(self) -> BaseException | None:
        return self.__cause__

    def to_log(self) -> NamedDict:
        return {
            "msg": self.message,
            "extra": self.extra,
        }


class AstroDatabaseError(AstroError):
    """Raised when an error occurs relating to the database."""

    def __init__(
        self,
        *,
        message: str,
        db_path: StrPath | None = None,
        db_exists: bool | None = None,
        db_type: str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = extra or {}

        if db_path is not None:
            extra["db_path"] = str(db_path)
            extra["db_name"] = Path(db_path).name

        if db_exists is not None:
            extra["db_exists"] = db_exists

        if db_type is not None:
            extra["db_type"] = db_type

        super().__init__(message=message, extra=extra, caused_by=caused_by)


class SetupError(AstroError):
    """Raised when an error occurs during setup."""

    def __init__(
        self,
        *,
        cause: str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Construct parent
        super().__init__(
            message=f"Setup error occurred: {cause}", extra=extra, caused_by=caused_by
        )


class ExpectedVariableType(AstroError):
    """Raised when a variable's type doesn't match the expected type(s).

    This error provides detailed information about type validation failures,
    including the variable name, actual type received, and expected types.
    """

    def __init__(
        self,
        *,
        var_name: str,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
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
        expected_str = options_to_str(expected_types)
        message = f"Expected {var_name} to be {expected_str}. Got {got_type} instead"

        if with_value is not None:
            extra["got_value"] = with_value
            message += f" with value {with_value}"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class ExpectedElementTypeError(AstroError):
    """Raised when an element's type within a collection doesn't match the expected type(s).

    This error provides detailed information about type validation failures for elements,
    including the collection name, actual type received, and expected types.
    """

    def __init__(
        self,
        *,
        collection_name: str,
        expected: Sequence[type] | type,
        got: type,
        index_or_key: int | Any | None = None,
        with_value: Any | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
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
            "collection_name": collection_name,
            "got_type": got_type,
            "expected_types": expected_types,
        }

        # Error message
        expected_str = options_to_str(expected_types)
        message = f"Expected elements of {collection_name} to be {expected_str}. Got {got_type} instead"

        if index_or_key is not None:
            extra["index_or_key"] = index_or_key
            message += f" at index/key {index_or_key}"

        if with_value is not None:
            extra["got_value"] = with_value
            message += f" with value {with_value}"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class ExpectedTypeError(AstroError):
    """Raised when a type doesn't match the expected type(s) without variable context.

    This is the base class for type validation errors that don't involve specific variables.
    """

    def __init__(
        self,
        *,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
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
        expected_str = options_to_str(expected_types)
        message = f"Expected type to be {expected_str}. Got {got_type} instead"

        if with_value is not None:
            extra["got_value"] = with_value
            message += f" with value {with_value}"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class KeyTypeError(ExpectedVariableType):
    """Raised when a key's type doesn't match the expected type(s)."""

    def __init__(
        self,
        *,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        super().__init__(
            var_name="key",
            expected=expected,
            got=got,
            with_value=with_value,
            caused_by=caused_by,
            extra=extra,
        )


class KeyStrError(KeyTypeError):
    """Raised when a key is expected to be a string but isn't."""

    def __init__(
        self,
        *,
        got: type,
        with_value: Any | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        super().__init__(
            expected=str,
            got=got,
            with_value=with_value,
            extra=extra,
            caused_by=caused_by,
        )


class ValueTypeError(ExpectedVariableType):
    """Raised when a value's type doesn't match the expected type(s)."""

    def __init__(
        self,
        *,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        super().__init__(
            var_name="value",
            expected=expected,
            got=got,
            with_value=with_value,
            caused_by=caused_by,
            extra=extra,
        )


class EmptyStructureError(AstroError):
    """Raised when a structure is expected to have elements but is empty.

    This error provides detailed information about the empty structure,
    including its name, type, and length.
    """

    def __init__(
        self,
        *,
        structure_name: str,
        structure_type: type[Collection[Any] | Mapping[Any, Any]],
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = (extra or {}) | {
            "structure_name": structure_name,
            "structure_type": structure_type,
        }

        # Error message
        message = (
            f"Expected {structure_name!r} of type {structure_type.__name__} "
            "to have elements, but it is empty"
        )
        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class NoEntryError(AstroError, KeyError):
    def __init__(
        self,
        *,
        key_value: Any,
        sources: Sequence[str] | str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = (extra or {}) | {"key_value": key_value}

        # Normalize sources to a sequence for consistent processing
        sources_seq: Sequence[str] = []
        if sources is not None:
            if isinstance(sources, Sequence) and not isinstance(sources, str):
                sources_seq = sources
            else:
                sources_seq = [sources]
            extra["sources"] = sources_seq

            # Prepare formatted source descriptions for message
            formatted_sources = []
            for source in sources_seq:
                formatted_sources.append(repr(source))
        else:
            formatted_sources = []

        # Error message
        message = f"No entry for key {key_value}"
        if formatted_sources:
            message += " in sources " + options_to_str(formatted_sources)

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class LoadError(AstroError):
    """Raised when an error occurs when loading."""

    def __init__(
        self,
        *,
        path_or_uid: StrPath | int | None = None,
        obj_or_key: Any | None = None,
        load_from: str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
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
        message += f" {type_name(obj_or_key)} to"

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
            message += f" {str(path_or_uid)}"

        # Add uid
        elif path_or_uid is not None:
            extra["uid"] = path_or_uid
            message += f" using uid {secretify(path_or_uid)}"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class SaveError(AstroError):
    """Raised when an error occurs when saving."""

    def __init__(
        self,
        *,
        path_or_uid: StrPath | None = None,
        obj_to_save: Any | None = None,
        save_to: str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
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
        message += f" {type_name(obj_to_save)} to"

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
            message += f" {str(path_or_uid)}"

        # Add uid
        elif path_or_uid is not None:
            extra["uid"] = path_or_uid
            message += f" using uid {secretify(path_or_uid)}"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class CreationError(AstroError):
    """Raised when an error occurs during creation of an object."""

    def __init__(
        self,
        *,
        object_type: type | str,
        reason: str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = extra or {}
        extra["object_type"] = object_type

        # Error message
        object_type_name = (
            object_type if isinstance(object_type, str) else type_name(object_type)
        )
        message = f"Error occurred during creation of {object_type_name}"
        if reason is not None and len(reason.strip()) > 0:
            extra["reason"] = reason
            message += f": {reason}"
        else:
            extra["reason"] = "unspecified"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)

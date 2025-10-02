# --- Internal Imports ---
from abc import ABC
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

# --- External Imports ---
from sqlalchemy.exc import IntegrityError  # noqa: F401

# --- Local Imports ---
from astro.typings import (
    HashableObject,
    ImmutableRecord,
    ModelName,
    ModelProvider,
    NamedDict,
    RecordableModel,
    StrPath,
    get_path_type,
    options_to_str,
    secretify,
    type_name,
    type_options,
)

AstroErrorType = TypeVar("AstroErrorType", bound="AstroError")
PythonErrorType = TypeVar("PythonErrorType", bound=Exception)

if TYPE_CHECKING:
    from astro.llms.base import LLMConfig


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


class RecordableIdentityError(AstroError):
    """When the identity of two models are not the same when they should be."""

    def __init__(
        self,
        *,
        record: HashableObject,
        other_record: HashableObject,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
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
        super().__init__(message=message, extra=extra, caused_by=caused_by)


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


class NoEntryError(AstroError, KeyError):
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
        caused_by: Exception | None = None,
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
                else:
                    formatted_sources.append(repr(src))
            if object_hashes:
                extra["object_hash"] = (
                    object_hashes if len(object_hashes) > 1 else object_hashes[0]
                )
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
        obj_or_key: RecordableModel | ImmutableRecord | Any | None = None,
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
        obj_to_save: RecordableModel | ImmutableRecord | Any | None = None,
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
        message += f" {type_name(obj_to_save)}"

        # Object has hash
        if isinstance(obj_to_save, (RecordableModel, ImmutableRecord)):
            object_hash = (
                hash(obj_to_save)
                if isinstance(obj_to_save, RecordableModel)
                else obj_to_save.record_hash
            )
            extra["object_hash"] = object_hash
            message += f" ({secretify(object_hash)})"

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
            message += f" {str(path_or_uid)}"

        # Add uid
        elif path_or_uid is not None:
            extra["uid"] = path_or_uid
            message += f" using uid {secretify(path_or_uid)}"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class ModelFileStoreError(AstroError):
    """Raised when an error occurs with a store operation."""

    from astro.paths import ModelFileStore

    def __init__(
        self,
        *,
        operation: str,
        reason: str | None = None,
        stores: ModelFileStore | Sequence[ModelFileStore] | None = None,
        model_or_uid: RecordableModel | type[RecordableModel] | str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = extra or {}
        extra["operation"] = operation

        # Error message
        message = f"Error occurred during store operation `{operation}`"

        if model_or_uid is not None and isinstance(model_or_uid, RecordableModel):
            extra["model_type"] = type(model_or_uid)
            extra["model_hash"] = hash(model_or_uid)
            message += (
                f" of model {type_name(model_or_uid)} ({secretify(hash(model_or_uid))})"
            )
        if (
            model_or_uid is not None
            and isinstance(model_or_uid, type)
            and issubclass(model_or_uid, RecordableModel)
        ):
            extra["model_type"] = model_or_uid
            message += f" of model type {model_or_uid.__name__}"
        elif model_or_uid is not None:
            extra["model_uid"] = model_or_uid
            message += f" for model uid {secretify(model_or_uid)}"

        if stores is not None and isinstance(stores, Sequence):
            for store in stores:
                name = store.name.lower() if store.name else "unnamed"
                extra[f"store_{name}_name"] = store.name
                extra[f"store_{name}_root_dir"] = store.root_dir
                extra[f"store_{name}_model_type"] = store.model_type
                extra[f"store_{name}_index_path"] = store.index_file
            message += f" in stores {options_to_str([store.name for store in stores])}"
        elif stores is not None:
            extra["store_name"] = stores.name
            extra["store_root_dir"] = stores.root_dir
            extra["store_model_type"] = stores.model_type
            extra["store_index_path"] = stores.index_file
            message += (
                f" in store {stores.name} for model type {stores.model_type.__name__}"
            )

        if reason is not None and len(reason.strip()) > 0:
            extra["reason"] = reason
            message += f": {reason}"
        else:
            extra["reason"] = "unspecified"

        # Construct parent
        super().__init__(message=message, extra=extra, caused_by=caused_by)


class LLMError(AstroError):
    """Raised when an error occurs relating to LLMs."""

    def __init__(
        self,
        *,
        message: str,
        model_identifier: str | ModelName | ModelProvider | None = None,
        model_name: str | ModelName | ModelProvider | Any = None,
        model_provider: str | ModelProvider | Any = None,
        model_config: "LLMConfig | None" = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = extra or {}

        if model_identifier is not None:
            extra["model_identifier"] = model_identifier

        if model_name is not None:
            extra["model_name"] = model_name

        if model_provider is not None:
            extra["model_provider"] = model_provider

        if model_config is not None:
            extra["model_config"] = model_config.model_dump(mode="json")
            extra["model_config_type"] = type(model_config)
            extra["model_config_hash"] = hash(model_config)

        super().__init__(message=message, extra=extra, caused_by=caused_by)


class SupportError(AstroError):
    """Raised when something is not supported by Astro."""

    def __init__(
        self,
        *,
        feature: str,
        reason: str | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = extra or {}
        extra["feature"] = feature

        # Error message
        message = f"'{feature}' is not supported by Astro"
        if reason is not None and len(reason.strip()) > 0:
            extra["reason"] = reason
            message += f": {reason}"
        else:
            extra["reason"] = "unspecified"

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


# --- SQL Errors ---


class SQLIntegrityError(AstroDatabaseError):
    """Raised when a database integrity error occurs."""

    def __init__(
        self,
        *,
        operation: str,
        reason: str | None = None,
        database: str | None = None,
        table: str
        | type[ImmutableRecord[Any]]
        | Sequence[type[ImmutableRecord[Any]]]
        | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = extra or {}
        extra["operation"] = operation

        if database is not None:
            extra["database"] = database

        # Error message
        message = f"Database integrity error during {operation} operation"

        if table is not None:
            extra["table"] = table
            if isinstance(table, Sequence) and not isinstance(table, str):
                message += " on tables " + options_to_str(
                    [str(subtable.__tablename__) for subtable in table]
                )
            elif isinstance(table, type):
                message += f" on table {table.__tablename__}"
            else:
                message += f" on table {table}"

        if reason is not None and len(reason.strip()) > 0:
            extra["reason"] = reason
            message += f": {reason}"
        else:
            extra["reason"] = "unspecified"

        # Construct parent
        super().__init__(
            message=message,
            db_path=database,
            db_type="SQL",
            extra=extra,
            caused_by=caused_by,
        )


class SQLRetrievalError(AstroDatabaseError):
    """Raised when a database retrieval error occurs."""

    def __init__(
        self,
        *,
        operation: str,
        reason: str | None = None,
        database: str | None = None,
        table: str
        | type[ImmutableRecord[Any]]
        | Sequence[type[ImmutableRecord[Any]]]
        | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ):
        # Error details
        extra = extra or {}
        extra["operation"] = operation

        if database is not None:
            extra["database"] = database

        # Error message
        message = f"Database retrieval error during {operation} operation"

        if table is not None:
            extra["table"] = table
            if isinstance(table, Sequence) and not isinstance(table, str):
                message += " on tables " + options_to_str(
                    [str(subtable.__tablename__) for subtable in table]
                )
            elif isinstance(table, type):
                message += f" on table {table.__tablename__}"
            else:
                message += f" on table {table}"

        if reason is not None and len(reason.strip()) > 0:
            extra["reason"] = reason
            message += f": {reason}"
        else:
            extra["reason"] = "unspecified"

        # Construct parent
        super().__init__(
            message=message,
            db_path=database,
            db_type="SQL",
            extra=extra,
            caused_by=caused_by,
        )


if __name__ == "__main__":
    from astro.typings import ImmutableRecord, RecordableModel

    class TestModel(RecordableModel, frozen=True):
        id: int
        name: str

    obj = TestModel(id=1, name="Test")

    try:
        1 / 0  # type: ignore
    except Exception as error:
        raise SaveError(
            path_or_uid="/home/brian/PhD/astro/.gitignore",
            save_to="file",
            obj_to_save=obj,
            caused_by=error,
        )

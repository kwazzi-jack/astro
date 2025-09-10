# astro/logger.py

import datetime
import enum
import json
import logging
import os
import platform
import socket
import traceback
from collections.abc import Sequence
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.theme import Theme

from astro.errors import (
    AstroError,
    AstroErrorType,
    ExpectedElementTypeError,
    ExpectedTypeError,
    ExpectedVarType,
    KeyStrError,
    KeyTypeError,
    LoadError,
    ModelFileStoreError,
    NoEntryError,
    PythonErrorType,
    RecordableIdentityError,
    SaveError,
    SetupError,
    ValueTypeError,
)
from astro.typings import (
    HashableObject,
    ImmutableRecord,
    NamedDict,
    RecordableModel,
    StrPath,
    type_name,
)

# --- GLOBALS ---
_LOGGY_TAG_ATTR = "_from_loggy"


class LogLevel(enum.IntEnum):
    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG

    @classmethod
    def from_str(cls, value: str) -> "LogLevel":
        match value.upper():
            case "CRITICAL":
                return LogLevel.CRITICAL
            case "ERROR":
                return LogLevel.ERROR
            case "WARNING":
                return LogLevel.WARNING
            case "INFO":
                return LogLevel.INFO
            case "DEBUG":
                return LogLevel.DEBUG
            case _:
                raise ValueError(f"Unknown log level: {value}")


# --- Configuration ---
_BASE_LOG_LEVEL = logging.INFO


def _get_log_file():
    """Get the log file path, importing _LOG_DIR lazily to avoid circular imports."""
    from astro.paths import _LOG_DIR

    if _LOG_DIR is None:
        return "astro.jsonl"
    return _LOG_DIR / "astro.jsonl"


# Internal flag to ensure setup runs only once
_setup_done = False

# Define custom theme for Rich
_CUSTOM_THEME = Theme(
    {
        "logging.level.debug": "bold green",
        "logging.level.info": "bold cyan",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "debug": "dim blue",
        "logging.time": "bold blue",
    }
)

# Local Timezone
_LOCAL_TZ = datetime.datetime.now().astimezone().tzinfo


class JsonFormatter(logging.Formatter):
    """Custom formatter to output logs in JSONL format."""

    def __init__(self):
        super().__init__()
        self.standard_attributes = {
            "args",
            "asctime",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "module",
            "msecs",
            "message",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "thread",
            "threadName",
        }

        self.system_context = {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pid": os.getpid(),
            "application": "astro",
        }

    def format(self, record):
        """Formats a log record into a JSON string with extended context."""
        log_data = {
            "timestamp": self._format_timestamp(record.created),
            "name": record.name,
            "level": record.levelname,
            "levelno": record.levelno,
            "filename": record.filename,
            "lineno": record.lineno,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "pathname": record.pathname,
            "process": record.process,
            "thread": record.thread,
            "system": self.system_context,
        }

        if record.exc_info:
            log_data["exception"] = self._format_exception(record.exc_info)
        if record.stack_info:
            log_data["stack_info"] = self._format_stack(record.stack_info)

        extra = self._get_extra_fields(record)
        if extra:
            log_data["extra"] = extra

        return json.dumps(log_data, ensure_ascii=False)

    def _format_exception(self, exc_info):
        """Formats exception information into a string."""
        return "".join(traceback.format_exception(*exc_info))

    def _format_stack(self, stack_info):
        """Formats stack information (already a string)."""
        return stack_info

    def _get_extra_fields(self, record):
        """Extracts extra fields from the log record."""
        return {
            key: value
            for key, value in record.__dict__.items()
            if key not in self.standard_attributes
        }

    def _format_timestamp(self, created):
        """Formats the timestamp in ISO 8601 UTC format."""
        return datetime.datetime.fromtimestamp(created, tz=_LOCAL_TZ).isoformat()


def dt_formatter(dt: datetime.datetime) -> Text:
    return Text(dt.isoformat(sep=" ", timespec="milliseconds"))


def _get_module_logger(filepath: StrPath) -> logging.Logger:
    # Convert str to Path
    if isinstance(filepath, str):
        filepath = Path(filepath)

    # Extract package level traversal
    logger_name = filepath.stem
    for part in reversed(filepath.parts[:-1]):
        logger_name = f"{part}.{logger_name}"
        if part == "astro":
            break

    # Send back logger with package name
    return logging.getLogger(logger_name)


class Loggy:
    """A logging wrapper that provides context-aware logging and error handling.

    This class wraps the standard Python `logging` module to provide a simpler
    interface. It provides helper methods for common built-in Python exceptions
    and all custom exceptions derived from `astro.errors.BaseError`, allowing
    them to be created and logged in a single step.

    Attributes:
        `_module_path` (`StrPath`): The file path of the module using this logger instance.
        `_logger` (`logging.Logger`): The underlying logger instance.
    """

    # Lazy references
    from astro.paths import ModelFileStore

    def __init__(self, module_path: StrPath) -> None:
        """Initializes the Loggy instance.

        Args:
            `module_path` (`StrPath`): The path to the module where the logger is used.
        """
        self._module_path = module_path
        self._logger = _get_module_logger(module_path)

    def info(
        self,
        message: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level `INFO` on the logger.

        Args:
            `message` (`str`): The message to be logged.
            `execution_info` (`bool`, optional): If `True`, exception info is added to the log. Defaults to `False`.
            `stack_info` (`bool`, optional): If `True`, stack info is added to the log. Defaults to `False`.
            `extra` (`Any`, optional): Additional context to include in the log. Defaults to `None`.
        """
        self._logger.info(
            message, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def debug(
        self,
        message: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level `DEBUG` on the logger.

        Args:
            `message` (`str`): The message to be logged.
            `execution_info` (`bool`, optional): If `True`, exception info is added to the log. Defaults to `False`.
            `stack_info` (`bool`, optional): If `True`, stack info is added to the log. Defaults to `False`.
            `extra` (`Any`, optional): Additional context to include in the log. Defaults to `None`.
        """
        self._logger.debug(
            message, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def warning(
        self,
        message: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level `WARNING` on the logger.

        Args:
            `message` (`str`): The message to be logged.
            `execution_info` (`bool`, optional): If `True`, exception info is added to the log. Defaults to `False`.
            `stack_info` (`bool`, optional): If `True`, stack info is added to the log. Defaults to `False`.
            `extra` (`Any`, optional): Additional context to include in the log. Defaults to `None`.
        """
        self._logger.warning(
            message, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def error(
        self,
        message: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level `ERROR` on the logger.

        Args:
            `message` (`str`): The message to be logged.
            `execution_info` (`bool`, optional): If `True`, exception info is added to the log. Defaults to `False`.
            `stack_info` (`bool`, optional): If `True`, stack info is added to the log. Defaults to `False`.
            `extra` (`Any`, optional): Additional context to include in the log. Defaults to `None`.
        """
        self._logger.error(
            message, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def critical(
        self,
        message: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level `CRITICAL` on the logger.

        Args:
            `message` (`str`): The message to be logged.
            `execution_info` (`bool`, optional): If `True`, exception info is added to the log. Defaults to `False`.
            `stack_info` (`bool`, optional): If `True`, stack info is added to the log. Defaults to `False`.
            `extra` (`Any`, optional): Additional context to include in the log. Defaults to `None`.
        """
        self._logger.critical(
            message, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def exception(
        self,
        msg_or_obj: str | Any,
        execution_info: bool = True,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level `ERROR` and exception info.

        This method should be called from an exception handler.

        Args:
            `msg_or_obj` (`str | Any`): The message or object to be logged.
            `execution_info` (`bool`, optional): If `True`, exception info is added to the log. Defaults to `True`.
            `stack_info` (`bool`, optional): If `True`, stack info is added to the log. Defaults to `False`.
            `extra` (`Any`, optional): Additional context to include in the log. Defaults to `None`.
        """
        self._logger.exception(
            msg_or_obj, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def _log_builtin_exception(self, error: Exception):
        """Log a built-in exception with appropriate error handling.

        Attempts to log the exception with full exception info if currently in an
        exception handler context. Falls back to a simple error log if exception
        logging fails.

        Args:
            `error` (`Exception`): The built-in exception to log.
        """
        message = f"{type_name(error)}: {error}"
        try:
            # Try log exception if in exception handler
            self.exception(message, execution_info=True, extra={})
        except Exception:
            # Fallback to error log
            self.error(message, extra={})

    def _log_and_return_builtin_error(
        self,
        *,
        error: PythonErrorType,
        extra: NamedDict | None = None,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
    ) -> PythonErrorType:
        """Logs a built-in error's details and returns the error instance.

        Args:
            `error` (`PythonErrorType`): The built-in error to log and return.
            `extra` (`NamedDict | None`, optional): Additional details to include in the log for built-in errors. Defaults to `None`.
            `caused_by` (`AstroError | Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.

        Returns:
            `PythonErrorType`: The created error instance.

        Notes:
            If `caused_by` is provided and of type `AstroError` or tagged by `Loggy`, it will not be logged again.
        """
        # Skip logging if specified
        if skip_error_log:
            return error

        # Check if the error is already tagged as from Loggy
        is_loggy_error = getattr(error, _LOGGY_TAG_ATTR, False)

        # Wrapping another exception (Ignore if already a BaseError or tagged)
        if (
            caused_by is not None
            and not is_loggy_error
            and isinstance(caused_by, Exception)
        ):
            self._log_builtin_exception(caused_by)

        # Log Python error
        self.error(message=f"{type_name(error)}: {error}", extra=extra or {})

        # Tag the error as coming from Loggy
        setattr(error, _LOGGY_TAG_ATTR, True)

        # Return the original error
        return error

    def _log_and_return_astro_error(
        self,
        *,
        error: AstroErrorType,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
    ) -> AstroErrorType:
        """Logs the error's details and returns the error instance.

        Args:
            `error` (`AstroErrorType`): The astro error to log and return.
            `caused_by` (`AstroError | Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.

        Returns:
            `AstroErrorType`: The created error instance.

        Notes:
            If `caused_by` is provided and of type `AstroError` or tagged by `Loggy`, it will not be logged again.
        """
        # Skip logging if specified
        if skip_error_log:
            return error

        # Check if the error is already tagged as from Loggy
        is_loggy_error = getattr(error, _LOGGY_TAG_ATTR, False)

        # Wrapping another exception (Ignore if already a BaseError or tagged)
        if (
            caused_by is not None
            and not is_loggy_error
            and isinstance(caused_by, Exception)
        ):
            self._log_builtin_exception(caused_by)

        # Log error message
        self.error(**error.to_log())

        # Tag the error as coming from Loggy
        setattr(error, _LOGGY_TAG_ATTR, True)

        # Return the original error
        return error

    # --- Built-in Error Logging Methods ---
    def ValueError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ValueError:
        """Creates, logs, and returns a `ValueError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `ValueError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=ValueError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def TypeError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> TypeError:
        """Creates, logs, and returns a `TypeError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `TypeError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=TypeError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def KeyError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> KeyError:
        """Creates, logs, and returns a `KeyError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `KeyError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=KeyError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def FileNotFoundError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> FileNotFoundError:
        """Creates, logs, and returns a `FileNotFoundError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `FileNotFoundError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=FileNotFoundError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def AttributeError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> AttributeError:
        """Creates, logs, and returns an `AttributeError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `AttributeError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=AttributeError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def IndexError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> IndexError:
        """Creates, logs, and returns an `IndexError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `IndexError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=IndexError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def IOError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> IOError:
        """Creates, logs, and returns an `IOError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `IOError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=IOError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def OSError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> OSError:
        """Creates, logs, and returns an `OSError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `OSError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=OSError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def RuntimeError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> RuntimeError:
        """Creates, logs, and returns a `RuntimeError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `RuntimeError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=RuntimeError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def NotImplementedError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> NotImplementedError:
        """Creates, logs, and returns a `NotImplementedError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `NotImplementedError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=NotImplementedError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def FileExistsError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> FileExistsError:
        """Creates, logs, and returns a `FileExistsError`.

        Args:
            `message` (`str`): The error message.
            `caused_by` (`AstroError | Exception | None`, optional): The original exception that caused this error. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `**extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `FileExistsError`: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=FileExistsError(message),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    # --- Astro Error Logging ---

    def SetupError(
        self,
        cause: str,
        caused_by: AstroError | Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> SetupError:
        """Creates, logs, and returns a `SetupError` from an existing exception.

        Args:
            `cause` (`str`): The cause of the error.
            `caused_by` (`AstroError | Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `SetupError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=SetupError(cause=cause, extra=extra),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def RecordableIdentityError(
        self,
        *,
        record: HashableObject,
        other_record: HashableObject,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> RecordableIdentityError:
        """Creates, logs, and returns a `RecordableIdentityError`.

        Args:
            `record` (`HashableObject`): The first record.
            `other_record` (`HashableObject`): The second record.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `RecordableIdentityError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=RecordableIdentityError(
                record=record, other_record=other_record, extra=extra
            ),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def ExpectedVarType(
        self,
        *,
        var_name: str,
        got: type,
        expected: Sequence[type] | type,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ExpectedVarType:
        """Creates, logs, and returns an `ExpectedVarType`.

        Args:
            `var_name` (`str`): The name of the variable.
            `got` (`type`): The type received.
            `expected` (`Sequence[type] | type`): The type(s) expected.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `ExpectedVarType`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=ExpectedVarType(
                var_name=var_name, got=got, expected=expected, extra=extra
            ),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def ExpectedElementTypeError(
        self,
        *,
        collection_var_name: str,
        got: type,
        expected: Sequence[type] | type,
        index_or_key: int | Any | None = None,
        extra: NamedDict | None = None,
        caused_by: Exception | None = None,
    ) -> ExpectedElementTypeError:
        """Creates, logs, and returns an `ExpectedElementTypeError`.

        Args:
            `collection_var_name` (`str`): The name of the collection variable.
            `got` (`type`): The type received.
            `expected` (`Sequence[type] | type`): The type(s) expected.
            `index_or_key` (`int | Any | None`, optional): The index or key where the error occurred. Defaults to `None`.
            `extra` (`NamedDict | None`, optional): Additional details to include in the log. Defaults to `None`.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.

        Returns:
            `ExpectedElementTypeError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=ExpectedElementTypeError(
                collection_var_name=collection_var_name,
                got=got,
                expected=expected,
                index_or_key=index_or_key,
                extra=extra,
            ),
            caused_by=caused_by,
            skip_error_log=False,
        )

    def ExpectedTypeError(
        self,
        *,
        got: type,
        expected: Sequence[type] | type,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ExpectedTypeError:
        """Creates, logs, and returns an `ExpectedTypeError`.

        Args:
            `got` (`type`): The type received.
            `expected` (`Sequence[type] | type`): The type(s) expected.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `ExpectedTypeError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=ExpectedTypeError(got=got, expected=expected, extra=extra),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def KeyTypeError(
        self,
        *,
        got: type,
        expected: Sequence[type] | type,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> KeyTypeError:
        """Creates, logs, and returns a `KeyTypeError`.

        Args:
            `got` (`type`): The type received.
            `expected` (`Sequence[type] | type`): The type(s) expected.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `KeyTypeError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=KeyTypeError(got=got, expected=expected, extra=extra),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def KeyStrError(
        self,
        *,
        got: type,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> KeyStrError:
        """Creates, logs, and returns a `KeyStrError`.

        Args:
            `got` (`type`): The type received.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `KeyStrError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=KeyStrError(got=got, extra=extra),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def ValueTypeError(
        self,
        *,
        got: type,
        expected: Sequence[type] | type,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ValueTypeError:
        """Creates, logs, and returns a `ValueTypeError`.

        Args:
            `got` (`type`): The type received.
            `expected` (`Sequence[type] | type`): The type(s) expected.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `ValueTypeError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=ValueTypeError(got=got, expected=expected, extra=extra),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def NoEntryError(
        self,
        *,
        key_value: Any,
        sources: Sequence[str | RecordableModel | ImmutableRecord]
        | str
        | RecordableModel
        | ImmutableRecord
        | None = None,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> NoEntryError:
        """Creates, logs, and returns a `NoEntryError`.

        Args:
            `key_value` (`Any`): The key that was not found.
            `sources` (`Sequence[str | RecordableModel | ImmutableRecord] | str | RecordableModel | ImmutableRecord | None`, optional): The source(s) that were searched. Defaults to `None`.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `NoEntryKeyError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=NoEntryError(key_value=key_value, sources=sources, extra=extra),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def LoadError(
        self,
        *,
        path_or_uid: StrPath | int | None = None,
        obj_or_key: RecordableModel | ImmutableRecord | Any | None = None,
        load_from: str | None = None,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> LoadError:
        """Creates, logs, and returns a `LoadError`.

        Args:
            `path_or_uid` (`StrPath | int | None`, optional): The path or UID of the object that failed to load. Defaults to `None`.
            `obj_or_key` (`RecordableModel | ImmutableRecord | Any | None`, optional): The object or key that failed to load. Defaults to `None`.
            `load_from` (`str | None`, optional): The source from which the object was being loaded. Defaults to `None`.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `LoadError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=LoadError(
                path_or_uid=path_or_uid,
                obj_or_key=obj_or_key,
                load_from=load_from,
                extra=extra,
            ),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def SaveError(
        self,
        *,
        path_or_uid: StrPath | None = None,
        obj_to_save: RecordableModel | ImmutableRecord | Any | None = None,
        save_to: str | None = None,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> SaveError:
        """Creates, logs, and returns a `SaveError`.

        Args:
            `path_or_uid` (`StrPath | None`, optional): The path or UID of the object that failed to save. Defaults to `None`.
            `obj_to_save` (`RecordableModel | ImmutableRecord | Any | None`, optional): The object that failed to save. Defaults to `None`.
            `save_to` (`str | None`, optional): The destination where the object was being saved. Defaults to `None`.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `SaveError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=SaveError(
                path_or_uid=path_or_uid,
                obj_to_save=obj_to_save,
                save_to=save_to,
                extra=extra,
            ),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def ModelFileStoreError(
        self,
        *,
        operation: str,
        reason: str | None = None,
        stores: ModelFileStore | Sequence[ModelFileStore] | None = None,
        model_or_uid: RecordableModel | type[RecordableModel] | str | None = None,
        caused_by: Exception | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ModelFileStoreError:
        """Creates, logs, and returns a `ModelFileStoreError`.

        Args:
            `operation` (`str`): The operation that failed.
            `reason` (`str | None`, optional): The reason for the failure. Defaults to `None`.
            `store` (`ModelFileStore | Sequence[ModelFileStore] | None`, optional): The name of the store. Defaults to `None`.
            `model_or_uid` (`RecordableModel | type[RecordableModel] | str | None`, optional): The model involved in the operation. Defaults to `None`.
            `caused_by` (`Exception | None`, optional): The error to wrap. Defaults to `None`.
            `skip_error_log` (`bool`, optional): If `True`, logging is skipped. Defaults to `False`.
            `extra` (`Any`, optional): Additional details to include in the log.

        Returns:
            `ModelFileStoreError`: The created error instance.
        """
        return self._log_and_return_astro_error(
            error=ModelFileStoreError(
                operation=operation,
                reason=reason,
                stores=stores,
                model_or_uid=model_or_uid,
                extra=extra,
            ),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )


def setup_logging():
    """Configures the base `astro` logger with handlers."""
    global _setup_done
    if _setup_done:
        return

    base_logger = logging.getLogger("astro")
    base_logger.setLevel(logging.DEBUG)

    # Rich handler for console
    console = Console(theme=_CUSTOM_THEME)
    rich_handler = RichHandler(
        console=console,
        level=_BASE_LOG_LEVEL,
        show_time=True,
        show_level=True,
        show_path=True,
        log_time_format=dt_formatter,
        omit_repeated_times=False,
        enable_link_path=True,
        rich_tracebacks=True,
        markup=True,
    )

    # JSONL handler for file
    file_handler = TimedRotatingFileHandler(
        filename=_get_log_file(),
        when="midnight",
        interval=1,
        backupCount=14,  # Two weeks of logs
        encoding="utf-8",
        delay=False,
    )
    file_handler.setFormatter(JsonFormatter())
    file_handler.setLevel(logging.DEBUG)

    if not base_logger.handlers:
        base_logger.addHandler(rich_handler)
        base_logger.addHandler(file_handler)

    _setup_done = True


def get_loggy(filepath: StrPath) -> "Loggy":
    """Retrieves a configured logger instance for a given module.

    This function ensures that logging is set up before returning a `Loggy`
    instance, which is tailored to the module from which it is called.

    Args:
        `filepath` (`StrPath`): The file path of the module, typically `__file__`.

    Returns:
        `Loggy`: A configured `Loggy` instance.
    """
    setup_logging()
    return Loggy(filepath)


# Initialize logging when module is imported
setup_logging()

if __name__ == "__main__":
    from pathlib import Path

    from astro.paths import _LOG_DIR

    print(_LOG_DIR)
    main_logger = get_loggy(__file__)
    util_logger = get_loggy(Path(__file__).parent / "util.py")

    main_logger.debug("Debug message (not shown)")
    main_logger.info("Info message")
    util_logger.warning("Warning message")
    util_logger.error("Error message")

    try:
        1 / 0  # type: ignore
    except ZeroDivisionError:
        main_logger.exception("Exception occurred")

    main_logger.critical("Get help. We are melting")
    print(f"\nCheck console output and JSONL logs at '{_get_log_file()}'")

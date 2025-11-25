# astro/logger.py
"""Logging helpers and Loggy error utilities for Astro."""

# --- Internal Imports ---
import logging
from collections.abc import Collection, Mapping, Sequence
from enum import IntEnum
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

# --- Local Imports ---
from astro.errors import (
    AstroError,
    AstroErrorType,
    CreationError,
    EmptyStructureError,
    ExpectedElementTypeError,
    ExpectedTypeError,
    ExpectedVariableType,
    KeyStrError,
    KeyTypeError,
    LoadError,
    NoEntryError,
    ParseError,
    PythonErrorType,
    SaveError,
    SetupError,
    ValueTypeError,
)
from astro.typings import (
    NamedDict,
    StrPath,
    type_name,
)

# --- GLOBALS ---
_LOGGY_TAG_ATTR = "_from_loggy"


class LogLevel(IntEnum):
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


def _get_log_file() -> Path | str:
    """Get the log file path, importing _LOG_DIR lazily to avoid circular imports."""
    from astro.paths import LOG_DIR

    if LOG_DIR is None:
        return "astro.log"
    return LOG_DIR / "astro.log"


class _ExtraFormatter(logging.Formatter):
    """Custom formatter that appends extra fields to log messages.

    Automatically formats extra keyword arguments as key=value pairs
    separated by pipes at the end of the log message.
    """

    # Standard LogRecord attributes that should not be treated as extras
    _RESERVED_ATTRS = {
        "name",
        "msg",
        "args",
        "created",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "thread",
        "threadName",
        "exc_info",
        "exc_text",
        "stack_info",
        "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record, appending any extra fields.

        Args:
            record: The log record to format.

        Returns:
            str: The formatted log message with extras appended.
        """
        # Get the base formatted message
        base_message = super().format(record)

        # Extract extra fields (any attribute not in the reserved set)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in self._RESERVED_ATTRS
        }

        # If there are extras, append them to the message
        if extras:
            extra_parts = [f"{key}={value}" for key, value in extras.items()]
            return f"{base_message} | {', '.join(extra_parts)}"

        return base_message


# Internal flag to ensure setup runs only once
_setup_done = False

logging.getLogger("astro").addHandler(logging.NullHandler())


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

    This class wraps the standard Python logging module to provide a simpler
    interface. It provides helper methods for common built-in Python exceptions
    and all custom exceptions derived from astro.errors.BaseError, allowing
    them to be created and logged in a single step.

    Attributes:
        _module_path (StrPath): The file path of the module using this logger instance.
        _logger (logging.Logger): The underlying logger instance.
    """

    def __init__(self, module_path: StrPath) -> None:
        """Initializes the Loggy instance.

        Args:
            module_path (StrPath): The path to the module where the logger is used.
        """
        self._module_path = module_path
        self._logger = _get_module_logger(module_path)

    def info(
        self,
        msg: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level INFO on the logger.

        Args:
            msg (str): The message to be logged.
            execution_info (bool, optional): If True, exception info is added to the log. Defaults to False.
            stack_info (bool, optional): If True, stack info is added to the log. Defaults to False.
            extra (Any, optional): Additional context to include in the log. Defaults to None.
        """
        self._logger.info(
            msg, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def debug(
        self,
        msg: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level DEBUG on the logger.

        Args:
            msg (str): The message to be logged.
            execution_info (bool, optional): If True, exception info is added to the log. Defaults to False.
            stack_info (bool, optional): If True, stack info is added to the log. Defaults to False.
            extra (Any, optional): Additional context to include in the log. Defaults to None.
        """
        self._logger.debug(
            msg, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def warning(
        self,
        msg: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level WARNING on the logger.

        Args:
            msg (str): The message to be logged.
            execution_info (bool, optional): If True, exception info is added to the log. Defaults to False.
            stack_info (bool, optional): If True, stack info is added to the log. Defaults to False.
            extra (Any, optional): Additional context to include in the log. Defaults to None.
        """
        self._logger.warning(
            msg, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def error(
        self,
        msg: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level ERROR on the logger.

        Args:
            msg (str): The message to be logged.
            execution_info (bool, optional): If True, exception info is added to the log. Defaults to False.
            stack_info (bool, optional): If True, stack info is added to the log. Defaults to False.
            extra (Any, optional): Additional context to include in the log. Defaults to None.
        """
        self._logger.error(
            msg, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def critical(
        self,
        msg: str,
        execution_info: bool = False,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level CRITICAL on the logger.

        Args:
            message (str): The message to be logged.
            execution_info (bool, optional): If True, exception info is added to the log. Defaults to False.
            stack_info (bool, optional): If True, stack info is added to the log. Defaults to False.
            extra (Any, optional): Additional context to include in the log. Defaults to None.
        """
        self._logger.critical(
            msg, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def exception(
        self,
        msg: str | Any,
        execution_info: bool = True,
        stack_info: bool = False,
        **extra: Any,
    ):
        """Logs a message with level ERROR and exception info.

        This method should be called from an exception handler.

        Args:
            msg (str | Any): The message or object to be logged.
            execution_info (bool, optional): If True, exception info is added to the log. Defaults to True.
            stack_info (bool, optional): If True, stack info is added to the log. Defaults to False.
            extra (Any, optional): Additional context to include in the log. Defaults to None.
        """
        self._logger.exception(
            msg, exc_info=execution_info, stack_info=stack_info, extra=extra
        )

    def _log_builtin_exception(self, error: Exception):
        """Log a built-in exception with appropriate error handling.

        Attempts to log the exception with full exception info if currently in an
        exception handler context. Falls back to a simple error log if exception
        logging fails.

        Args:
            error (Exception): The built-in exception to log.
        """
        message = f"{type_name(error)}: {error}"
        try:
            # Try log exception if in exception handler
            self.exception(message, execution_info=True)
        except Exception:
            # Fallback to error log
            self.error(message)

    def _log_and_return_builtin_error(
        self,
        *,
        error: PythonErrorType,
        extra: NamedDict | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
    ) -> PythonErrorType:
        """Logs a built-in error's details and returns the error instance.

        Args:
            error (PythonErrorType): The built-in error to log and return.
            extra (NamedDict | None, optional): Additional details to include in the log for built-in errors. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.

        Returns:
            PythonErrorType: The created error instance.

        Notes:
            If caused_by is provided and of type AstroError or tagged by Loggy, it will not be logged again.
        """
        # Skip logging if specified
        if skip_error_log:
            return error

        # Log warning if provided
        if warning is not None:
            self.warning(warning)

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
        self.error(msg=f"{type_name(error)}: {error}", extra=extra or {})

        # Tag the error as coming from Loggy
        setattr(error, _LOGGY_TAG_ATTR, True)

        # Return the original error
        return error

    def _log_and_return_astro_error(
        self,
        *,
        error: AstroErrorType,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
    ) -> AstroErrorType:
        """Logs the error's details and returns the error instance.

        Args:
            error (AstroErrorType): The astro error to log and return.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.

        Returns:
            AstroErrorType: The created error instance.

        Notes:
            If caused_by is provided and of type AstroError or tagged by Loggy, it will not be logged again.
        """
        # Skip logging if specified
        if skip_error_log:
            return error

        # Log warning if provided
        if warning is not None:
            self.warning(warning)

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
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ValueError:
        """Creates, logs, and returns a ValueError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            ValueError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=ValueError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def TypeError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> TypeError:
        """Creates, logs, and returns a TypeError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            TypeError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=TypeError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def KeyError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> KeyError:
        """Creates, logs, and returns a KeyError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            KeyError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=KeyError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def FileNotFoundError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> FileNotFoundError:
        """Creates, logs, and returns a FileNotFoundError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            FileNotFoundError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=FileNotFoundError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def AttributeError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> AttributeError:
        """Creates, logs, and returns an AttributeError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            AttributeError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=AttributeError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def IndexError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> IndexError:
        """Creates, logs, and returns an IndexError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            IndexError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=IndexError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def IOError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> IOError:
        """Creates, logs, and returns an IOError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            IOError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=IOError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def OSError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> OSError:
        """Creates, logs, and returns an OSError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            OSError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=OSError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def RuntimeError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> RuntimeError:
        """Creates, logs, and returns a RuntimeError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            RuntimeError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=RuntimeError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def NotImplementedError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> NotImplementedError:
        """Creates, logs, and returns a NotImplementedError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            NotImplementedError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=NotImplementedError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    def FileExistsError(
        self,
        message: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> FileExistsError:
        """Creates, logs, and returns a FileExistsError.

        Args:
            message (str): The error message.
            caused_by (AstroError | Exception | None, optional): The original exception that caused this error. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            **extra (Any, optional): Additional details to include in the log.

        Returns:
            FileExistsError: The created error instance.
        """
        return self._log_and_return_builtin_error(
            error=FileExistsError(message),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
            extra=extra,
        )

    # --- Local Error Logging ---

    def SetupError(
        self,
        cause: str,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> SetupError:
        """Creates, logs, and returns a SetupError from an existing exception.

        Args:
            cause (str): The cause of the error.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            SetupError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=SetupError(cause=cause, extra=extra),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def ExpectedVariableType(
        self,
        *,
        var_name: str,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ExpectedVariableType:
        """Creates, logs, and returns an ExpectedVarType.

        Args:
            var_name (str): The name of the variable.
            expected (Sequence[type] | type): The type(s) expected.
            got (type): The type received.
            with_value (Any | None): The actual value received, if available. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            ExpectedVariableType: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=ExpectedVariableType(
                var_name=var_name,
                expected=expected,
                got=got,
                with_value=with_value,
                extra=extra,
                caused_by=caused_by,
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def ExpectedElementTypeError(
        self,
        *,
        structure_var_name: str,
        expected: Sequence[type] | type,
        got: type,
        index_or_key: int | Any | None = None,
        with_value: Any | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ExpectedElementTypeError:
        """Creates, logs, and returns an ExpectedElementTypeError.

        Args:
            collection_var_name (str): The name of the collection variable.
            expected (Sequence[type] | type): The type(s) expected.
            got (type): The type received.
            index_or_key (int | Any | None, optional): The index or key where the error occurred. Defaults to None.
            with_value (Any | None, optional): The actual value received, if available. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            ExpectedElementTypeError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=ExpectedElementTypeError(
                collection_name=structure_var_name,
                expected=expected,
                got=got,
                index_or_key=index_or_key,
                with_value=with_value,
                extra=extra,
                caused_by=caused_by,
            ),
            caused_by=caused_by,
            skip_error_log=skip_error_log,
        )

    def ExpectedTypeError(
        self,
        *,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ExpectedTypeError:
        """Creates, logs, and returns an ExpectedTypeError.

        Args:
            expected (Sequence[type] | type): The type(s) expected.
            got (type): The type received.
            with_value (Any | None, optional): The actual value received, if available. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            ExpectedTypeError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=ExpectedTypeError(
                expected=expected, got=got, with_value=with_value, extra=extra
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def EmptyStructureError(
        self,
        *,
        structure_name: str,
        structure_type: type[Collection[Any] | Mapping[Any, Any]],
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> EmptyStructureError:
        """
        Creates and returns an EmptyStructureError for an empty structure.

        Args:
            structure_name (str): The name of the empty structure.
            structure_type (type[Collection[Any] | Mapping[Any, Any]]): The type of the structure (e.g., list, dict).
            caused_by (AstroError | Exception | None, optional): The exception that caused this error. Defaults to None.
            warning (str | None, optional): A warning message to log. Defaults to None.
            skip_error_log (bool, optional): Whether to skip logging the error. Defaults to False.
            extra (Any, optional): Additional extra data to include in the error.

        Returns:
            EmptyStructureError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=EmptyStructureError(
                structure_name=structure_name,
                structure_type=structure_type,
                extra=extra,
                caused_by=caused_by,
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def KeyTypeError(
        self,
        *,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> KeyTypeError:
        """Creates, logs, and returns a KeyTypeError.

        Args:
            expected (Sequence[type] | type): The type(s) expected.
            got (type): The type received.
            with_value (Any | None, optional): The actual value received, if available. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            KeyTypeError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=KeyTypeError(
                expected=expected, got=got, with_value=with_value, extra=extra
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def KeyStrError(
        self,
        *,
        got: type,
        with_value: Any | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> KeyStrError:
        """Creates, logs, and returns a KeyStrError.

        Args:
            got (type): The type received.
            with_value (Any | None, optional): The actual value received, if available. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            KeyStrError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=KeyStrError(got=got, with_value=with_value, extra=extra),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def ValueTypeError(
        self,
        *,
        expected: Sequence[type] | type,
        got: type,
        with_value: Any | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ValueTypeError:
        """Creates, logs, and returns a ValueTypeError.

        Args:
            expected (Sequence[type] | type): The type(s) expected.
            got (type): The type received.
            with_value (Any | None, optional): The actual value received, if available. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            ValueTypeError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=ValueTypeError(
                expected=expected, got=got, with_value=with_value, extra=extra
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def NoEntryError(
        self,
        *,
        key_value: Any,
        sources: Sequence[str] | str | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> NoEntryError:
        """Creates, logs, and returns a NoEntryError.

        Args:
            key_value (Any): The key that was not found.
            sources (Sequence[str] | str | None, optional): The source(s) that were searched. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            NoEntryKeyError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=NoEntryError(key_value=key_value, sources=sources, extra=extra),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def LoadError(
        self,
        *,
        path_or_uid: StrPath | int | None = None,
        obj_or_key: Any | None = None,
        load_from: str | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> LoadError:
        """Creates, logs, and returns a LoadError.

        Args:
            path_or_uid (StrPath | int | None, optional): The path or UID of the object that failed to load. Defaults to None.
            obj_or_key (Any | None, optional): The object or key that failed to load. Defaults to None.
            load_from (str | None, optional): The source from which the object was being loaded. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            LoadError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=LoadError(
                path_or_uid=path_or_uid,
                obj_or_key=obj_or_key,
                load_from=load_from,
                extra=extra,
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def SaveError(
        self,
        *,
        path_or_uid: StrPath | None = None,
        obj_to_save: Any | None = None,
        save_to: str | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> SaveError:
        """Creates, logs, and returns a SaveError.

        Args:
            path_or_uid (StrPath | None, optional): The path or UID of the object that failed to save. Defaults to None.
            obj_to_save (Any | None, optional): The object that failed to save. Defaults to None.
            save_to (str | None, optional): The destination where the object was being saved. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            SaveError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=SaveError(
                path_or_uid=path_or_uid,
                obj_to_save=obj_to_save,
                save_to=save_to,
                extra=extra,
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def CreationError(
        self,
        *,
        object_type: type | str,
        reason: str | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> CreationError:
        """Creates, logs, and returns a CreationError.

        Args:
            object_type (type | str): The type of object that failed to be created.
            reason (str | None, optional): The reason why the creation failed. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            CreationError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=CreationError(
                object_type=object_type, reason=reason, extra=extra, caused_by=caused_by
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )

    def ParseError(
        self,
        *,
        type_to_parse: type,
        value_to_parse: Any,
        expected_type: type,
        reason: str | None = None,
        caused_by: AstroError | Exception | None = None,
        warning: str | None = None,
        skip_error_log: bool = False,
        **extra: Any,
    ) -> ParseError:
        """Creates, logs, and returns a ParseError.

        Args:
            type_to_parse (type): The type being parsed from.
            value_to_parse (Any): The value being parsed.
            expected_type (type): The type expected after parsing.
            reason (str | None, optional): The reason why parsing failed. Defaults to None.
            caused_by (AstroError | Exception | None, optional): The error to wrap. Defaults to None.
            warning (str | None, optional): An optional warning message to log before the error. Defaults to None.
            skip_error_log (bool, optional): If True, logging is skipped. Defaults to False.
            extra (Any, optional): Additional details to include in the log.

        Returns:
            ParseError: The created error instance.
        """
        if warning is not None:
            self.warning(warning)

        return self._log_and_return_astro_error(
            error=ParseError(
                type_to_parse=type_to_parse,
                value_to_parse=value_to_parse,
                expected_type=expected_type,
                reason=reason,
                extra=extra,
                caused_by=caused_by,
            ),
            caused_by=caused_by,
            warning=warning,
            skip_error_log=skip_error_log,
        )


def setup_logging(level: str = "DEBUG", log_file: str | None = None) -> None:
    """Configure the astro logger with a file handler.

    Args:
        level (str): Desired logging level for the astro logger. Defaults to "DEBUG".
        log_file (str | None, optional): Explicit log file path. Defaults to astro.log.

    Raises:
        ValueError: If the provided logging level is invalid.

    Examples:
        >>> setup_logging(level="DEBUG")
    """

    global _setup_done

    base_logger = logging.getLogger("astro")

    if _setup_done or any(
        getattr(handler, "_astro_managed", False) for handler in base_logger.handlers
    ):
        base_logger.setLevel(LogLevel.from_str(level).value)
        _setup_done = True
        return

    log_path: str | Path
    if log_file is not None:
        log_path = log_file
    else:
        log_path = _get_log_file()

    handler = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",
        interval=1,
        backupCount=14,
        encoding="utf-8",
        delay=False,
    )
    handler.setFormatter(
        _ExtraFormatter(
            fmt="%(asctime)s %(name)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S%z",
        )
    )
    handler.setLevel(logging.DEBUG)
    handler._astro_managed = True  # type: ignore[attr-defined]

    base_logger.addHandler(handler)
    base_logger.setLevel(LogLevel.from_str(level).value)

    # Checkpoint log
    base_logger.debug("Checkpoint -- logging setup complete")

    _setup_done = True


def get_loggy(filepath: StrPath) -> "Loggy":
    """Retrieves a configured logger instance for a given module.

    This function ensures that logging is set up before returning a Loggy
    instance, which is tailored to the module from which it is called.

    Args:
        filepath (StrPath): The file path of the module, typically __file__.

    Returns:
        Loggy: A configured Loggy instance.
    """
    loggy = Loggy(filepath)
    loggy.debug("Module imported and logging initialised")
    return loggy

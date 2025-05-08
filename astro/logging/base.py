# astro/logger.py

import logging
from logging.handlers import TimedRotatingFileHandler
import json
import os
import platform
import socket
import traceback
import datetime

from rich.logging import RichHandler
from rich.theme import Theme
from rich.console import Console


from astro.paths import LOG_DIR

# --- Configuration ---
_BASE_LOG_LEVEL = logging.INFO
_LOG_FILE = LOG_DIR / "astro.jsonl"  # JSONL file for structured logs

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


def dt_formatter(dt: datetime.datetime) -> str:
    return dt.isoformat(sep=" ", timespec="milliseconds")


def setup_logging():
    """Configures the base `astro` logger with handlers."""
    global _setup_done
    if _setup_done:
        return

    base_logger = logging.getLogger("astro")
    base_logger.setLevel(_BASE_LOG_LEVEL)

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
        filename=_LOG_FILE,
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


def get_logger(name: str) -> logging.Logger:
    """Retrieves a configured logger instance."""
    setup_logging()
    logger_name = f"astro.{name}"
    return logging.getLogger(logger_name)


# Initialize logging when module is imported
setup_logging()

if __name__ == "__main__":
    print(LOG_DIR)
    main_logger = get_logger("main_test")
    util_logger = get_logger("utils_test")

    main_logger.debug("Debug message (not shown)")
    main_logger.info("Info message")
    util_logger.warning("Warning message")
    util_logger.error("Error message")

    try:
        1 / 0
    except ZeroDivisionError:
        main_logger.exception("Exception occurred")

    main_logger.critical("Get help. We are melting")
    print(f"\nCheck console output and JSONL logs at '{_LOG_FILE}'")

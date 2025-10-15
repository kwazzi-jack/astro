from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, computed_field

from astro.loggings import get_loggy
from astro.typings import NamedDict, StrDict, type_name
from astro.utilities.system import (
    get_platform_details,
    get_platform_str,
    get_python_details,
    get_python_environment_str,
)
from astro.utilities.timing import (
    get_date_str,
    get_datetime_now,
    get_datetime_str,
    get_day_period_str,
    get_time_str,
)

"""Module for managing various contexts used in LLM interactions.

This module provides classes to handle different types of context information
such as datetime, platform, and Python environment details. Contexts can be
configured to update live or remain static.
"""

# --- Globals ---
loggy = get_loggy(__file__)


class Context(BaseModel, ABC):
    """Base class for collective context objects aggregating multiple information sources.

    Provides a foundation for classes that combine various information types.
    """

    @classmethod
    def contains(cls, value: str) -> bool:
        if not isinstance(value, str):
            raise loggy.ExpectedTypeError(
                expected=str, got=type(value), with_value=value
            )
        return value in cls.model_fields or value in cls.model_computed_fields

    @abstractmethod
    def to_formatted(self) -> NamedDict:
        raise loggy.NotImplementedError("Subclasses of Context must implement this")


class NoneContext(Context):
    """Null case for Context type.

    Use to help with internal type checking for prompts that do not need context.
    """


class ChatContext(Context):
    """Main context class for chat interactions.

    Aggregates various information types (datetime, platform, Python) for LLM chats.
    Datetime information is live by default for current time, others are static.
    """

    @computed_field
    @property
    def datetime(self) -> datetime:
        return get_datetime_now()

    @computed_field
    @property
    def day_period(self) -> str:
        return get_day_period_str(self.datetime)

    def to_formatted(self) -> StrDict:
        dt = self.datetime
        datetime_str = get_datetime_str(dt)
        period_str = get_day_period_str(dt)
        return {"datetime": datetime_str, "day_period": period_str}


def select_context_type(cls_name: str) -> type[Context]:
    context_classes = {
        "none": NoneContext,
        "chat": ChatContext,
    }

    if cls_name not in context_classes:
        raise loggy.ValueError(f"Cannot find context type associated with {cls_name!r}")

    return context_classes[cls_name]


if __name__ == "__main__":
    ...

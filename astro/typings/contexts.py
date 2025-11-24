"""Base context classes for Astro's language model workflows."""

# --- Internal Imports ---
from abc import ABC, abstractmethod
from typing import TypeVar

# --- External Imports ---
from pydantic import BaseModel

# --- Local Imports ---
from astro.typings.base import type_name


# --- Base Context Class ---
class Context(BaseModel, ABC):
    """Base class for collective context objects aggregating multiple information sources.

    Provides a foundation for classes that combine various information types.
    """

    @classmethod
    def contains(cls, value: str) -> bool:
        if not isinstance(value, str):
            raise ValueError(
                f"Expected str. Got {type_name(value)} with value {value!r}"
            )
        return value in cls.model_fields or value in cls.model_computed_fields

    @abstractmethod
    def to_formatted(self):
        raise NotImplementedError("Subclasses of Context must implement this")


ContextType = TypeVar("ContextType", bound=Context)

__all__ = [
    "Context",
    "ContextType",
]

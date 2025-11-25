# --- Internal Imports ---
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any, TypeAlias, TypeAliasType, TypeVar

# --- External Imports ---
from prompt_toolkit import HTML
from pydantic_ai.tools import Tool, ToolFuncEither
from rich.markdown import Markdown

# --- Local Imports ---
from astro.typings.contexts import Context
from astro.typings.inputs import StreamIn
from astro.typings.outputs import StreamOut

# Functions
Fn: TypeAlias = Callable[..., Any]
AnyFn: TypeAlias = Callable[[], Any]
DateTimeFn: TypeAlias = Callable[[], datetime]
FloatFn: TypeAlias = Callable[[], float]
HTMLFn: TypeAlias = Callable[[], HTML]
InlineFn: TypeAlias = Callable[[], None]
MarkdownFn: TypeAlias = Callable[[], Markdown]
MarkupFn: TypeAlias = Callable[[], str]
StrFn: TypeAlias = Callable[[], str]
FnSequence: TypeAlias = Sequence[Fn]


# Agent Stream Types
StreamFn: TypeAlias = Callable[[StreamIn], StreamOut]

# Agent function types
ContextT = TypeVar("ContextT", bound=Context)

AgentFn = TypeAliasType(
    "AgentFn",
    Tool[ContextT] | ToolFuncEither[ContextT, ...],
    type_params=(ContextT,),
)
AgentFnSequence = TypeAliasType(
    "AgentFnSequence",
    Sequence[AgentFn[ContextT]],
    type_params=(ContextT,),
)

__all__ = [
    "Fn",
    "AgentFn",
    "AnyFn",
    "DateTimeFn",
    "FloatFn",
    "HTMLFn",
    "InlineFn",
    "MarkdownFn",
    "MarkupFn",
    "StrFn",
    "FnSequence",
    "AgentFnSequence",
    "StreamFn",
]

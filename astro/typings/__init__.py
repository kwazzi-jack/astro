"""Aggregated typing exports for Astro."""

# --- Internal Imports ---
from __future__ import annotations

# --- Local Imports (for exposing) ---
from astro.typings.base import (
    ArgsAdapter,
    HTMLDict,
    MessageList,
    NamedDict,
    PathDict,
    PathKind,
    PTKDecoration,
    PydanticEvent,
    StrDict,
    StrPath,
    get_path_type,
    literal_to_list,
    options_to_str,
    path_dict_to_str_dict,
    secretify,
    str_dict_to_path_dict,
    type_name,
    type_options,
)
from astro.typings.callables import (
    AnyFn,
    DateTimeFn,
    FloatFn,
    HTMLFn,
    InlineFn,
    MarkdownFn,
    MarkupFn,
    StreamFn,
    StrFn,
)
from astro.typings.inputs import StreamIn
from astro.typings.outputs import (
    AgentOutput,
    AgentOutputType,
    AgentText,
    AgentThink,
    AgentToolCall,
    AgentToolReturn,
    delta_to_agent_output,
    part_to_agent_output,
)

# --- Expose objects ---
__all__ = [
    "AgentOutput",
    "AgentOutputType",
    "ArgsAdapter",
    "AnyFn",
    "StreamFn",
    "DateTimeFn",
    "FloatFn",
    "HTMLDict",
    "HTMLFn",
    "InlineFn",
    "MarkdownFn",
    "MarkupFn",
    "MessageList",
    "NamedDict",
    "PTKDecoration",
    "PathDict",
    "PathKind",
    "PydanticEvent",
    "StreamIn",
    "StrDict",
    "StrPath",
    "StrFn",
    "AgentText",
    "AgentThink",
    "AgentToolCall",
    "AgentToolReturn",
    "delta_to_agent_output",
    "get_path_type",
    "literal_to_list",
    "options_to_str",
    "part_to_agent_output",
    "path_dict_to_str_dict",
    "secretify",
    "str_dict_to_path_dict",
    "type_name",
    "type_options",
]

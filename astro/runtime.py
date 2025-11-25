"""Convenience runtime helpers for embedding user code into Astro's agent.

Primarily this module provides `run_astro_with` which accepts a mixture of
callable objects, python file paths, or code snippets and exposes them to the
chat agent as tools. The goal is to keep the API minimal and ergonomic for
library users.
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import util as import_util
from pathlib import Path
from types import ModuleType
from typing import Any

from astro._bootstrap import _run_astro_cli
from astro.typings.callables import AgentFn


def _load_module_from_path(path: str | Path) -> ModuleType:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find file: {path}")
    spec = import_util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {path}")
    module = import_util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _extract_tools(items: Sequence[Any]) -> list[AgentFn]:
    """Turn caller-supplied items into agent-compatible tool functions.

    Supported inputs:
    - callable objects (functions)
    - path-like strings pointing to python files that export callables
    - module objects with callables

    Returns a list of callables suitable for passing as `tools` to the
    agent factory.
    """

    tools: list[AgentFn] = []
    for item in items:
        # Simple callable -> use directly
        if callable(item):
            tools.append(item)  # type: ignore[arg-type]
            continue

        # Module object -> harvest callables from attributes
        if isinstance(item, ModuleType):
            for attr_name in dir(item):
                attr = getattr(item, attr_name)
                if callable(attr) and not attr_name.startswith("_"):
                    tools.append(attr)  # type: ignore[arg-type]
            continue

        # Path-like string -> load module
        try:
            path_obj = Path(item)
        except Exception:
            raise TypeError(f"Unsupported tool item: {item!r}")

        module = _load_module_from_path(path_obj)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.startswith("_"):
                tools.append(attr)  # type: ignore[arg-type]

    return tools


def run_astro_with(
    items: Sequence[Any] | Any,
    *,
    instructions: str | Sequence[str] | None = None,
    overwrite_state: bool = False,
) -> None:
    """Convenience entrypoint to run Astro with injected tools.

    Args:
        items: A callable, module, path or sequence of such items to expose as
            tools to the agent.
        instructions: Optional extra instruction blocks appended to the
            system prompt.
        overwrite_state: Passed to the CLI builder when initialising state.
    """

    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        items_seq = (items,)
    else:
        items_seq = items

    tools = _extract_tools(items_seq)
    # Build and run
    _run_astro_cli(
        overwrite_state=overwrite_state, tools=tools, instructions=instructions
    )


__all__ = ["run_astro_with"]

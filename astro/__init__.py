"""Public API for the Astro library.

Expose a small, ergonomic surface that library users can import to run the
interactive CLI or embed Astro into their applications. The goal is to keep
the top-level API minimal while delegating implementation to internal
helpers in `_bootstrap.py` and `runtime.py`.
"""

from ._bootstrap import _setup_astro_cli as setup, _build_astro_cli as build_cli, _run_astro_cli as run_cli
from .runtime import run_astro_with

__all__ = ["setup", "build_cli", "run_cli", "run_astro_with"]
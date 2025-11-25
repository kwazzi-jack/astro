"""Bootstrap helpers for Astro's CLI entrypoints."""

from collections.abc import Sequence

from astro.contexts import ChatContext
from astro.typings.callables import AgentFn, AgentFnSequence


def _setup_astro_cli() -> None:
    """Configure shared paths, logging, and API configuration."""

    # Setup paths
    from astro.paths import setup_paths

    setup_paths()

    # Setup logging
    from astro.logger import setup_logging

    setup_logging()

    # Setup configs
    from astro.config import setup_api_config

    setup_api_config()


def _build_astro_cli(
    *,
    overwrite_state: bool = False,
    tools: AgentFn[ChatContext]
    | AgentFnSequence[ChatContext]
    | None = None,
    instructions: str | Sequence[str] | None = None,
):
    """Internal helper to build a configured AstroCLI instance."""

    from astro.app.cli import AstroCLI

    _setup_astro_cli()
    return AstroCLI(
        overwrite_state=overwrite_state,
        tools=tools,
        instructions=instructions,
    )


def _run_astro_cli(
    *,
    overwrite_state: bool = False,
    tools: AgentFn[ChatContext]
    | AgentFnSequence[ChatContext]
    | None = None,
    instructions: str | Sequence[str] | None = None,
) -> None:
    """Setup the runtime environment and launch AstroCLI."""

    # Setup environment and run
    cli = _build_astro_cli(
        overwrite_state=overwrite_state,
        tools=tools,
        instructions=instructions,
    )

    # Checkpoint log
    from astro.logger import get_loggy

    get_loggy(__file__).debug("Checkpoint -- starting AstroCLI (after setup)")

    cli.run()

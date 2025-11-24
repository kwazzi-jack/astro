import asyncio


def _setup_astro_cli() -> None:
    # Setup paths
    from astro.paths import setup_paths

    setup_paths()

    # Setup logging
    from astro.logger import setup_logging

    setup_logging()

    # Setup configs
    from astro.config import setup_api_config

    setup_api_config()


def _run_astro_cli() -> None:
    # Setup environment
    _setup_astro_cli()

    # Checkpoint log
    from astro.logger import get_loggy

    get_loggy(__file__).debug("Checkpoint -- starting AstroCLI (after setup)")

    # Run astro cli
    from astro.app.cli import AstroCLI

    AstroCLI().run()

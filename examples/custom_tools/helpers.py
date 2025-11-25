"""Helper functions registered as tools for the Astro CLI example."""

from __future__ import annotations

from datetime import datetime, timedelta

from observatory import Observatory, create_default_observatory

from astro.logger import get_loggy

_loggy = get_loggy(__file__)
_OBSERVATORY: Observatory = create_default_observatory()


def describe_site() -> str:
    """Return a paragraph describing the demo observatory.

    Returns:
        str: Readable description containing coordinates and tags.

    Examples:
        >>> describe_site().startswith("Astro Demo Station")
        True
    """

    return _OBSERVATORY.summary()


def list_instruments() -> str:
    """Return the available instruments with short annotations.

    Returns:
        str: Comma separated list of instrument descriptors.

    Examples:
        >>> "WideCam" in list_instruments()
        True
    """

    return _OBSERVATORY.instrument_table()


def schedule_observation(
    target: str,
    duration_minutes: int,
    *,
    exposures: int = 1,
) -> str:
    """Create a friendly observation plan for the requested target.

    Args:
        target (str): Target identifier or common name.
        duration_minutes (int): Total duration requested for the block.
        exposures (int): Number of exposures to split the block into.

    Returns:
        str: Summary describing when the block can run and how it is split.

    Raises:
        ValueError: Raised when duration_minutes or exposures are not positive.

    Examples:
        >>> schedule_observation("Vega", 30)
        'Scheduled Vega for 30 minutes (1 exposure) starting at ...'
    """

    if duration_minutes <= 0:
        raise _loggy.ValueError(
            "Duration must be a positive integer minute value",
            duration_minutes=duration_minutes,
        )
    if exposures <= 0:
        raise _loggy.ValueError(
            "Exposures must be positive",
            exposures=exposures,
        )

    now = datetime.utcnow()
    start_time = now + timedelta(minutes=5)
    per_exposure = duration_minutes / exposures
    return (
        f"Scheduled {target} for {duration_minutes} minutes "
        f"({exposures} exposure{'s' if exposures > 1 else ''}) starting at "
        f"{start_time:%Y-%m-%d %H:%M} UTC with {per_exposure:.1f} minute blocks."
    )


__all__ = [
    "describe_site",
    "list_instruments",
    "schedule_observation",
]

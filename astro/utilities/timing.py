"""Time and datetime utilities used across Astro."""

# --- Internal Imports ---
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from time import time_ns
from typing import TYPE_CHECKING, Literal

# --- Local Imports ---
from astro.typings.base import (
    literal_to_list,
    options_to_str,
)

if TYPE_CHECKING:
    from astro.typings.callables import FloatFn, InlineFn

# --- Globals ---
# Time Units and constants
TimeUnit = Literal["hour", "min", "sec", "msec", "microsec", "nanosec"]
_NANOSECONDS_CONVERSIONS = {
    "hour": 1e9 * 3600,
    "min": 1e9 * 60,
    "sec": 1e9,
    "msec": 1e6,
    "microsec": 1e3,
    "nanosec": 1.0,
}


# Datetime patterns and constants
DEFAULT_DATETIME_PATTERN = "%H:%M:%S.%f%z, %A, %d %B %Y"
DEFAULT_TIME_PATTERN = "%H:%M:%S.%f%z"
MORNING_END_HOUR = 12
AFTERNOON_END_HOUR = 17
DAY_END_HOUR = 24


def get_datetime_now(to_local: bool = True) -> datetime:
    """Return the current timezone-aware datetime.

    Args:
        to_local (bool, optional): Flag indicating whether to convert the result
            to the local timezone. Defaults to True.

    Returns:
        datetime: Current datetime in the requested timezone.

    Examples:
        >>> current = get_datetime_now()
        >>> current.tzinfo is not None
        True
    """
    return datetime.now().astimezone() if to_local else datetime.now(timezone.utc)


def _resolve_datetime(dt: datetime | None, to_local: bool) -> datetime:
    """Return a timezone-aware datetime suitable for formatting."""
    if dt is None:
        return get_datetime_now(to_local=to_local)
    return dt.astimezone() if to_local else dt


def get_time_str(
    pattern: str = DEFAULT_TIME_PATTERN,
    dt: datetime | None = None,
    to_local: bool = True,
) -> str:
    """Format a datetime into a time string.

    Args:
        pattern (str, optional): Time formatting pattern. Defaults to
            "%H:%M:%S.%f%z".
        dt (datetime | None, optional): Datetime to format. Defaults to the
            current time when None.
        to_local (bool, optional): Flag controlling local timezone conversion.
            Defaults to True.

    Returns:
        str: Formatted time string.

    Examples:
        >>> get_time_str(pattern="%H:%M")  # doctest: +SKIP
        '12:00'
    """
    current_dt = _resolve_datetime(dt, to_local)
    return current_dt.strftime(pattern)


def get_datetime_str(
    pattern: str = DEFAULT_DATETIME_PATTERN,
    dt: datetime | None = None,
    to_local: bool = True,
) -> str:
    """Format a datetime into a combined date and time string.

    Args:
        pattern (str, optional): Datetime formatting pattern. Defaults to
            "%H:%M:%S.%f%z, %A, %d %B %Y".
        dt (datetime | None, optional): Datetime to format. Defaults to the
            current datetime when None.
        to_local (bool, optional): Flag controlling local timezone conversion.
            Defaults to True.

    Returns:
        str: Formatted datetime string.

    Examples:
        >>> get_datetime_str(pattern="%Y-%m-%d")  # doctest: +SKIP
        '2024-01-01'
    """
    current_dt = _resolve_datetime(dt, to_local)
    return current_dt.strftime(pattern)


def get_day_period_str(dt: datetime | None = None, to_local: bool = True) -> str:
    """Return a readable period descriptor for a datetime.

    Args:
        dt (datetime | None, optional): Datetime to inspect. Defaults to the
            current datetime when None.
        to_local (bool, optional): Flag controlling local timezone conversion.
            Defaults to True.

    Returns:
        str: Human-readable period ("morning", "afternoon", or "evening").

    Raises:
        ValueError: Raised when the hour is outside the expected range.

    Examples:
        >>> get_day_period_str(get_datetime_now())
        'morning'  # doctest: +SKIP
    """
    current_dt = _resolve_datetime(dt, to_local)
    hour = current_dt.hour
    if 0 <= hour < MORNING_END_HOUR:
        return "morning"
    if MORNING_END_HOUR <= hour < AFTERNOON_END_HOUR:
        return "afternoon"
    if AFTERNOON_END_HOUR <= hour < DAY_END_HOUR:
        return "evening"
    raise ValueError(f"Unexpected hour value provided: {hour}")


def seconds_to_strtime(value: float) -> str:
    """Convert a duration in seconds to a compact textual representation.

    Args:
        value (float): Duration in seconds.

    Returns:
        str: Human-readable representation with up to two unit components.

    Examples:
        >>> seconds_to_strtime(5)
        '5.0s'
        >>> seconds_to_strtime(3600)
        '1h'
        >>> seconds_to_strtime(3725)
        '1h 2m'
    """
    absolute_value = abs(value)
    sign = "-" if value < 0 else ""

    units = [
        ("h", 3_600),
        ("m", 60),
        ("s", 1),
    ]

    parts: list[str] = []
    remaining = absolute_value

    for unit_name, unit_value in units:
        if remaining < unit_value:
            continue

        if unit_name == "s":
            count = remaining / unit_value
            parts.append(f"{sign}{count:.1f}{unit_name}")
            break

        count = remaining // unit_value
        parts.append(f"{sign}{int(count)}{unit_name}")
        remaining %= unit_value
        sign = ""

        if len(parts) == 2:
            break

    if not parts:
        return "<0.1s"

    return " ".join(parts)


def create_time_converter(
    input_unit: TimeUnit, output_unit: TimeUnit
) -> Callable[[float], float]:
    """Create a conversion function between two time units.

    Args:
        input_unit: Source time unit for the values that will be converted.
        output_unit: Target time unit for the converted values.

    Returns:
        Callable[[float], float]: Converter function that scales numeric values from
        input_unit to output_unit.

    Raises:
        ValueError: If an invalid input or output unit is provided.
    """

    # Input validation
    if input_unit not in _NANOSECONDS_CONVERSIONS:
        options = options_to_str(literal_to_list(TimeUnit), with_repr=True)
        raise ValueError(f"Invalid argument {input_unit=}. Choose from {options}")
    if output_unit not in _NANOSECONDS_CONVERSIONS:
        options = options_to_str(literal_to_list(TimeUnit), with_repr=True)
        raise ValueError(f"Invalid argument {output_unit=}. Choose from {options}")

    # Get conversation factor
    from_nano_factor = _NANOSECONDS_CONVERSIONS[input_unit]
    to_unit_factor = _NANOSECONDS_CONVERSIONS[output_unit]
    conv_factor = from_nano_factor / to_unit_factor

    # Create converter function
    def converter(value: float) -> float:
        return value * conv_factor

    return converter


def create_timer(unit: TimeUnit = "sec") -> tuple[InlineFn, FloatFn]:
    """Create a simple manual timer utility.

    Args:
        time_unit (TimeUnit, optional): Unit for the returned elapsed time.
            Defaults to "sec".
    Returns:
        (tuple[InlineFn, FloatFn]): Pair of callables
        where the first starts timing and the second stops timing, returning
        the elapsed seconds. Passing True to the stop callable resets internal
        state.

    Examples:
        >>> start, stop = create_timer()
        >>> start()
        >>> _ = stop()
    """

    # Create time converter function
    converter = create_time_converter("nanosec", unit)

    # Initial timer start value
    start_value = 0.0

    # Start function
    def start() -> None:
        """Starts the timer"""
        nonlocal start_value
        start_value = time_ns()

    # Stop function
    def stop() -> float:
        """Stops and resets the timer"""
        nonlocal start_value

        # start() not run
        if start_value == 0:
            return 0.0

        # Calculate total time taken
        elapsed = time_ns() - start_value

        # Result time and return
        start_value = 0.0
        return converter(elapsed)

    return start, stop

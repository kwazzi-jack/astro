import re
from collections.abc import Callable, Generator, Iterable
from datetime import datetime, timezone
from time import time

ASTRO_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f UTC%z"


def get_datetime_now(to_local: bool = True) -> datetime:
    """Get the current timezone-aware datetime.

    Args:
        to_local (bool): If True, return the current time in the local timezone.
            Otherwise, return time in UTC.

    Returns:
        datetime: Current datetime in the requested timezone.
    """
    return datetime.now().astimezone() if to_local else datetime.now(timezone.utc)


def get_timestamp(
    datet: datetime | None = None, pattern: str = ASTRO_DATETIME_FORMAT
) -> str:
    if datet is None:
        return get_datetime_now().strftime(pattern)
    else:
        return datet.strftime(pattern)


def from_timestamp(timestamp: str, pattern: str = ASTRO_DATETIME_FORMAT) -> datetime:
    try:
        dt = datetime.strptime(timestamp, pattern)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format. Got: {timestamp}") from e


def datetime_to_local(datet: datetime) -> datetime:
    return datet.astimezone()


def timestamp_to_local(timestamp: str, pattern: str = ASTRO_DATETIME_FORMAT) -> str:
    return get_timestamp(datetime_to_local(from_timestamp(timestamp, pattern)), pattern)


def get_date_str(
    pattern: str = "%A, %d %B %Y",
    dt: datetime | None = None,
    to_local: bool = True,
) -> str:
    if dt is None:
        current_dt = get_datetime_now(to_local=to_local)
    elif to_local:
        current_dt = datetime_to_local(dt)
    else:
        current_dt = dt
    return current_dt.strftime(pattern)


def get_time_str(
    pattern: str = "%H:%M:%S.%f%z",
    dt: datetime | None = None,
    to_local: bool = True,
) -> str:
    if dt is None:
        current_dt = get_datetime_now(to_local=to_local)
    elif to_local:
        current_dt = datetime_to_local(dt)
    else:
        current_dt = dt
    return current_dt.strftime(pattern)


def get_datetime_str(
    pattern: str = "%H:%M:%S.%f%z, %A, %d %B %Y",
    dt: datetime | None = None,
    to_local: bool = True,
) -> str:
    if dt is None:
        current_dt = get_datetime_now(to_local=to_local)
    elif to_local:
        current_dt = datetime_to_local(dt)
    else:
        current_dt = dt
    return current_dt.strftime(pattern)


def get_day_period_str(dt: datetime | None = None, to_local: bool = True) -> str:
    if dt is None:
        current_dt = get_datetime_now(to_local=to_local)
    elif to_local:
        current_dt = datetime_to_local(dt)
    else:
        current_dt = dt
    hour = current_dt.hour
    if 0 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 24:
        return "evening"
    else:
        raise ValueError(f"Unknown hour value: {hour}")


def strtime_to_seconds(time_str: str) -> float:
    """
    Convert a time string with unit to seconds.

    Parameters
    ----------
    time_str : str
        String representing time with unit (e.g., '2s', '10min', '1.5h')

    Returns
    -------
    float
        Time value converted to seconds

    Raises
    ------
    ValueError
        If the input string format is invalid or missing units

    Examples
    --------
    >>> strtime_to_seconds('2s')
    2.0
    >>> strtime_to_seconds('1.5h')
    5400.0
    >>> strtime_to_seconds('10min')
    600.0
    """
    if not isinstance(time_str, str):
        raise ValueError("Input must be a string")

    # Strip whitespace and convert to lowercase
    time_str = time_str.strip().lower()

    # Regular expression to match a number followed by a unit
    pattern = r"^(\d+\.?\d*)([a-z]+)$"
    match = re.match(pattern, time_str)

    if not match:
        raise ValueError(
            f"Invalid time format. Expected format: <number><unit> (e.g., '2s', '10min'). Got {time_str}"
        )

    value, unit = match.groups()

    # Convert value to float
    try:
        value = float(value)
    except ValueError:
        raise ValueError(f"Invalid numeric value: {value}")

    # Define unit conversion factors (to seconds)
    unit_map = {
        # Seconds
        "s": 1,
        "sec": 1,
        "second": 1,
        "seconds": 1,
        # Minutes
        "m": 60,
        "min": 60,
        "minute": 60,
        "minutes": 60,
        # Hours
        "h": 3600,
        "hr": 3600,
        "hour": 3600,
        "hours": 3600,
        # Days
        "d": 86400,
        "day": 86400,
        "days": 86400,
        # Weeks
        "w": 604800,
        "wk": 604800,
        "week": 604800,
        "weeks": 604800,
    }

    # Check if the unit is valid
    if unit not in unit_map:
        raise ValueError(
            f"Unknown time unit: '{unit}'. Supported units: {', '.join(unit_map.keys())}"
        )

    # Convert to seconds
    return value * unit_map[unit]


def seconds_to_strtime(value: float) -> str:
    """
    Convert a time value in seconds to a human-readable string with appropriate units.

    Returns up to two unit components for better readability (e.g., '2min 18.5s', '2h 20min').
    Seconds are displayed with one decimal place precision.

    Args:
        value: Time value in seconds

    Returns:
        str: Human-readable time string with up to two unit components

    Examples:
        >>> seconds_to_strtime(5)
        '0.5s'
        >>> seconds_to_strtime(3_600)
        '1h 0m'
        >>> nanoseconds_to_strtime(138)
        '2m 18.0s'
        >>> seconds_to_strtime(7_200)
        '2h 0m'
    """
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    # Define unit conversions in descending order (only hours, minutes, seconds)
    units = [
        ("h", 3_600),
        ("m", 60),
        ("s", 1),
    ]

    # Find the largest applicable unit
    parts = []
    remaining = abs_value

    for unit_name, unit_value in units:
        if remaining >= unit_value:
            if unit_name == "s":
                # For seconds, show one decimal place
                count = remaining / unit_value
                parts.append(f"{sign}{count:.1f}{unit_name}")
                remaining = 0
            else:
                # For hours and minutes, show whole numbers
                count = remaining // unit_value
                parts.append(f"{sign}{int(count)}{unit_name}")
                remaining = remaining % unit_value

            sign = ""  # Only apply sign to first component

    # If no parts were added (value is less than 1 second), return '0.0s'
    if not parts:
        return "<0.1s"

    return " ".join(parts)


def create_timer() -> tuple[Callable[[], None], Callable[[bool], float]]:
    start_value = 0

    def start():
        nonlocal start_value
        start_value = time()

    def stop(reset: bool) -> float:
        nonlocal start_value

        # No value to return
        if start_value == 0:
            return 0

        # Calculate time
        result = time() - start_value

        # Reset if set
        if reset:
            start_value = 0

        return result

    return start, stop

import re
from datetime import datetime, timezone

ASTRO_DATETIME_FORMAT = "%Y/%m/%d %H:%M:%S.%f UTC%z"


def get_datetime_now() -> datetime:
    return datetime.now().astimezone()


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


def get_date_str(dt: datetime) -> str:
    return dt.strftime("%A, %d %B %Y")


def get_time_str(dt: datetime) -> str:
    time_base = dt.strftime("%H:%M:%S.%f")[:-4]
    time_offset = dt.strftime("(%Z%z)")
    return f"{time_base} {time_offset}"


def get_datetime_str(dt: datetime) -> str:
    return f"{get_time_str(dt)}, {get_date_str(dt)}"


def get_day_period_str(dt: datetime) -> str:
    hour = dt.hour
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


if __name__ == "__main__":
    now = get_datetime_now()
    print(f"{get_date_str(now)=}")
    print(f"{get_time_str(now)=}")
    print(f"{get_day_period_str(now)=}")

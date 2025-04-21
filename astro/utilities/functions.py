from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Literal, Optional, get_args

import numpy as np
from numpy.typing import NDArray
from requests import request, RequestException
import yaml

ASTRO_DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"
LiteralType = Any


def get_datetime_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def get_timestamp(
    datet: Optional[datetime] = None, pattern: str = ASTRO_DATETIME_FORMAT
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


def is_api_alive(
    url: str,
    method: Literal["GET", "POST"] = "GET",
    headers: Optional[dict[str, str]] = None,
    timeout: Optional[float] = None,
    success_codes: Optional[tuple[int]] = None,
) -> bool:

    if url is None or len(url) == 0:
        raise ValueError("Empty or null URL for API check.")

    if headers is None or len(headers) == 0:
        headers = {}

    if timeout is None:
        timeout = 5
    elif timeout <= 0.0:
        raise ValueError(f"Only positive values allowed for timeout. Got {timeout=} ")

    if success_codes is None:
        success_codes = [200]
    elif len(success_codes) == 0:
        raise ValueError("No success codes provided for API check.")

    try:
        response = request(
            method=method,
            url=url,
            headers=headers,
            timeout=timeout,
        )
        return response.status_code in success_codes
    except (RequestException, ConnectionError, TimeoutError):
        return False


def round_up(number: float, ndigits: int = 0) -> int:
    offset = math.pow(10, ndigits)
    return round(math.ceil(number * offset) / offset)


def round_down(number: float, ndigits: int = 0) -> int:
    offset = math.pow(10, ndigits)
    return round(math.floor(number * offset) / offset)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r") as file:
        return yaml.safe_load(file)


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


def floor_binary_power(value: int | float) -> int:
    if not isinstance(value, (int, float)) or value <= 0:
        raise ValueError("Input must be a positive integer or float")

    if isinstance(value, float):
        value = int(value)

    return 1 << (value.bit_length() - 1)


def euclid_distance(
    x: int | float | np.number | NDArray[np.number],
    y: int | float | np.number | NDArray[np.number],
) -> float:
    return np.pow(x - y, 2)


def abs_distance(
    x: int | float | np.number | NDArray[np.number],
    y: int | float | np.number | NDArray[np.number],
) -> float:
    return np.abs(x - y)


def literal_to_tuple(literal: LiteralType) -> tuple[str, ...]:
    return get_args(literal)


if __name__ == "__main__":
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "config_store"
        / "models"
        / "llms"
        / "openai-gpt-4o-mini.yml"
    )
    print(f"{path.exists()=}")

    from pprint import pprint

    pprint(load_yaml(path), indent=4, depth=4)

import datetime as dt
from typing import Literal


def get_timestamp() -> str:
    """Get the current timestamp formatted as a localized datetime string.

    Returns:
        str: Current timestamp in format "DD/MM/YYYY, HH:MM:SS (TIMEZONE+OFFSET)"

    Examples:
        >>> timestamp = get_timestamp()
        >>> isinstance(timestamp, str)
        True
    """
    return (
        dt.datetime.now(dt.timezone.utc)
        .astimezone()
        .strftime("%d/%m/%Y, %H:%M:%S (%Z%z)")
    )


def get_time_of_day() -> Literal["morning", "afternoon", "evening"]:
    """Get the current time of day based on the hour.

    Returns:
        str: "morning" [0,12), "afternoon" [12,17), or "evening" [17,24)

    Raises:
        RuntimeError: If the hour is outside the expected 0-23 range

    Examples:
        >>> time_of_day = get_time_of_day()
        >>> time_of_day in ["morning", "afternoon", "evening"]
        True
    """
    now = dt.datetime.now()
    if 0 <= now.hour < 12:
        return "morning"
    elif 12 <= now.hour < 17:
        return "afternoon"
    elif 17 <= now.hour < 24:
        return "evening"
    else:
        # Something went wrong
        raise RuntimeError(f"Something went wrong: {now=}")


if __name__ == "__main__":
    print(f"{get_timestamp()=}")
    print(f"{get_time_of_day()=}")

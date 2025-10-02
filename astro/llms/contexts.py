from abc import ABC, abstractmethod

from astro.loggings import get_loggy
from astro.typings import StrDict, type_name
from astro.utilities.system import (
    get_platform_details,
    get_platform_str,
    get_python_details,
    get_python_environment_str,
)
from astro.utilities.timing import (
    get_datetime_now,
    get_datetime_str,
    get_day_period_str,
)

"""Module for managing various contexts used in LLM interactions.

This module provides classes to handle different types of context information
such as datetime, platform, and Python environment details. Contexts can be
configured to update live or remain static.
"""

# --- Globals ---
loggy = get_loggy(__file__)


class BaseInfo(ABC):
    """Base class for information objects.

    Provides a foundation for different information types with live or static update modes.

    Attributes:
        is_live: Whether the information updates live or remains static.
    """

    def __init__(self, *, live: bool = False):
        """Initialize the BaseInfo.

        Args:
            live: If True, information will update live; if False, remains static.
        """
        self._live = live
        loggy.debug(
            f"Created {type_name(self)} in {self.mode} mode",
        )
        self._str_cache: StrDict = {}

    def is_live(self) -> bool:
        """Whether the information updates live or remains static.

        Returns:
            bool: True if information updates live, False if it remains static.
        """
        return self._live

    def is_static(self) -> bool:
        """Whether the information updates live or remains static.

        Returns:
            bool: True if information remains static, False if it updates live.
        """
        return not self.is_live()

    @property
    def mode(self) -> str:
        """Get the mode of the information as a string.

        Returns:
            str: "live" if information updates live, "static" otherwise.
        """
        return "live" if self.is_live() else "static"

    def __repr__(self) -> str:
        """Return a string representation of the BaseInfo for debugging."""
        return f"{type_name(self)}(mode={self.mode!r})"

    def __str__(self) -> str:
        """Return a human-readable string representation of the BaseInfo."""
        return f"{type_name(self)} in {self.mode} mode"


class DateTimeInfo(BaseInfo):
    """Information for handling date and time details.

    Manages datetime data that can be updated live or kept static.
    Provides methods to get formatted datetime and day period strings.
    """

    def __init__(self, *, live: bool = False):
        """Initialize the DateTimeInfo.

        Sets the initial datetime value.

        Args:
            live (bool): If True, datetime will update on each access; if False, uses initial value.
        """
        super().__init__(live=live)
        self._datetime = get_datetime_now()

        # Cache initial strings if in static mode
        if self.is_static():
            self._str_cache["datetime"] = get_datetime_str(self._datetime)
            self._str_cache["day_period"] = get_day_period_str(self._datetime)
            loggy.debug(
                "Cached initial datetime strings for static mode", cache=self._str_cache
            )

    def get_datetime(self) -> str:
        """Get the current datetime as a formatted string.

        Updates the datetime if in live mode.

        Returns:
            str: Formatted datetime string.
        """
        # Live mode: update datetime on each access
        if self.is_live():
            self._datetime = get_datetime_now()
            return get_datetime_str(self._datetime)

        # Static mode: return cached string
        return self._str_cache["datetime"]

    def get_day_period(self) -> str:
        """Get the day period (morning, afternoon, etc.) as a string.

        Updates the datetime if in live mode.

        Returns:
            str: Day period string.
        """
        # Live mode: update datetime on each access
        if self.is_live():
            self._datetime = get_datetime_now()
            return get_day_period_str(self._datetime)

        # Static mode: return cached string
        return self._str_cache["day_period"]


class PlatformInfo(BaseInfo):
    """Information for handling platform details.

    Manages platform information that can be updated live or kept static.
    Provides methods to get formatted platform strings.
    """

    def __init__(self, *, live: bool = False):
        """Initialize the PlatformInfo.

        Sets the initial platform information.

        Args:
            live (bool): If True, platform information will update on each access; if False, uses initial value.
        """
        super().__init__(live=live)
        self._platform = get_platform_details()

        # Cache initial strings if in static mode
        if self.is_static():
            self._str_cache["platform"] = get_platform_str(self._platform)
            loggy.debug(
                "Cached initial platform strings for static mode", cache=self._str_cache
            )

    def get_platform(self) -> str:
        """Get the platform information as a formatted string.

        Updates platform information if in live mode.

        Returns:
            str: Formatted platform string.
        """
        # Live mode: update platform info on each access
        if self.is_live():
            self._platform = get_platform_details()
            return get_platform_str(self._platform)

        # Static mode: return cached string
        return self._str_cache["platform"]


class PythonInfo(BaseInfo):
    """Information for handling Python details.

    Manages Python information that can be updated live or kept static.
    Provides methods to get Python environment and version strings.
    """

    def __init__(self, *, live: bool = False):
        """Initialize the PythonInfo.

        Sets the initial Python information.

        Args:
            live: If True, Python information will update on each access; if False, uses initial value.
        """
        super().__init__(live=live)
        self._python = get_python_details()

        # Cache initial strings if in static mode
        if self.is_static():
            self._str_cache["python_env"] = get_python_environment_str(self._python)
            self._str_cache["python_version"] = self._python.version
            loggy.debug(
                "Cached initial python strings for static mode", cache=self._str_cache
            )

    def get_python_environment(self) -> str:
        """Get the Python environment information as a formatted string.

        Updates Python information if in live mode.

        Returns:
            str: Formatted Python environment string.
        """
        # Live mode: update Python info on each access
        if self.is_static():
            self._python = get_python_details()
            return get_python_environment_str(self._python)

        # Static mode: return cached string
        return self._str_cache["python_env"]

    def get_python_version(self) -> str:
        """Get the Python version as a string.

        Updates Python information if in live mode.

        Returns:
            str: Python version string.
        """
        # Live mode: update Python info on each access
        if self.is_static():
            self._python = get_python_details()
            return self._python.version

        # Static mode: return cached string
        return self._str_cache["python_version"]


class Context(ABC):
    """Base class for collective context objects aggregating multiple information sources.

    Provides a foundation for classes that combine various information types.
    """

    @abstractmethod
    def to_dict(self) -> StrDict:
        """Convert the context information to a dictionary.

        Returns:
            StrDict: Dictionary representation of the context information.
        """
        raise loggy.NotImplementedError(
            f"{type_name(self)} must implement to_dict method"
        )


class ChatContext(Context):
    """Main context class for chat interactions.

    Aggregates various information types (datetime, platform, Python) for LLM chats.
    Datetime information is live by default for current time, others are static.
    """

    def __init__(
        self,
        *,
        live: bool | None = None,
        datetime_live: bool = True,
        platform_live: bool = False,
        python_live: bool = False,
    ):
        """Initialize the ChatContext.

        Creates information sources with appropriate live/static modes.
        Datetime is live for current time, platform and Python are static.

        Args:
            live (bool, optional): If True, all information updates live; if False, all remain static; if None, use individual settings.
            datetime_live (bool): If True, datetime updates live; if False, remains static.
            platform_live (bool): If True, platform information updates live; if False, remains static.
            python_live (bool): If True, Python information updates live; if False, remains static.
        """
        super().__init__()

        # If live is set (True or False), use it for all; otherwise, use individual defaults
        live_is_not_none = live is not None
        self._datetime_info = DateTimeInfo(
            live=live if live_is_not_none else datetime_live
        )
        self._platform_info = PlatformInfo(
            live=live if live_is_not_none else platform_live
        )
        self._python_info = PythonInfo(live=live if live_is_not_none else python_live)

    def get_datetime(self) -> str:
        """Get the current datetime from the datetime context.

        Returns:
            str: Formatted datetime string.
        """
        return self._datetime_info.get_datetime()

    def get_day_period(self) -> str:
        """Get the day period from the datetime context.

        Returns:
            str: Day period string.
        """
        return self._datetime_info.get_day_period()

    def get_platform(self) -> str:
        """Get the platform details from the platform context.

        Returns:
            str: Formatted platform string.
        """
        return self._platform_info.get_platform()

    def get_python_environment(self) -> str:
        """Get the Python environment details from the Python context.

        Returns:
            str: Formatted Python environment string.
        """
        return self._python_info.get_python_environment()

    def get_python_version(self) -> str:
        """Get the Python version from the Python context.

        Returns:
            str: Python version string.
        """
        return self._python_info.get_python_version()

    def to_dict(self) -> StrDict:
        """Convert the chat context information to a dictionary.

        Returns:
            StrDict: Dictionary representation of the chat context information.
        """
        return {
            "datetime": self.get_datetime(),
            "day_period": self.get_day_period(),
            "platform": self.get_platform(),
            "python_environment": self.get_python_environment(),
            "python_version": self.get_python_version(),
        }

    def __repr__(self) -> str:
        """Return a detailed string representation of the ChatContext for debugging."""
        return (
            f"{type_name(self)} with "
            f"datetime={self._datetime_info!r}, "
            f"platform={self._platform_info!r}, "
            f"python={self._python_info!r}"
        )

    def __str__(self) -> str:
        """Return a human-readable string representation of the ChatContext."""
        context_values = [f"DateTime={self._datetime_info.mode}"]
        context_values.append(f"Platform={self._platform_info.mode}")
        context_values.append(f"Python={self._python_info.mode}")
        context_str = ",".join(context_values)
        return f"{type_name(self)}({context_str})"


if __name__ == "__main__":
    # Development test block to demonstrate ChatContext functionality
    from time import sleep

    context = ChatContext()
    print(context)
    print()
    print(repr(context))
    print()
    # Loop to show live datetime updates
    for i in range(10):
        print(f"======= Round {i + 1:02} =======")
        print(f"{context.get_datetime()=}")
        print(f"{context.get_day_period()=}")
        print(f"{context.get_platform()=}")
        print(f"{context.get_python_environment()=}")
        sleep(1)

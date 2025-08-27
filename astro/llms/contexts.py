from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from astro.utilities.system import (
    PlatformDetails,
    PythonEnvironmentDetails,
    get_platform_details,
    get_platform_str,
    get_python_environment_details,
    get_python_environment_str,
)
from astro.utilities.timing import (
    get_datetime_now,
    get_datetime_str,
    get_period_str,
)


class Context:
    def __init__(self, live: bool = False):
        self._live = live

    @property
    def is_live(self) -> bool:
        return self._live


class DateTimeContext(Context):
    def __init__(self, live: bool = False):
        super().__init__(live)
        self._datetime = get_datetime_now()

    def current_datetime(self) -> str:
        if self.is_live:
            self._datetime = get_datetime_now()
        return get_datetime_str(self._datetime)

    def current_period(self) -> str:
        if self.is_live:
            self._datetime = get_datetime_now()
        return get_period_str(self._datetime)


class PlatformContext(Context):
    def __init__(self, live: bool = False):
        super().__init__(live)
        self._platform = get_platform_details()

    def current_platform(self) -> str:
        if self.is_live:
            self._platform = get_platform_details()
        return get_platform_str(self._platform)


class PythonEnvironmentContext(Context):
    def __init__(self, live: bool = False):
        super().__init__(live)
        self._python_env = get_python_environment_details()

    def current_python_environment(self) -> str:
        if self.is_live:
            self._python_env = get_python_environment_details()
        return get_python_environment_str(self._python_env)

    def current_python_version(self) -> str:
        if self.is_live:
            self._python_env = get_python_environment_details()
        return self._python_env.version


class ChatContext:
    def __init__(self):
        self.datetime_ctx = DateTimeContext(live=True)  # Live updates
        self.platform_ctx = PlatformContext()
        self.python_ctx = PythonEnvironmentContext()

    def current_datetime(self) -> str:
        return self.datetime_ctx.current_datetime()

    def current_period(self) -> str:
        return self.datetime_ctx.current_period()

    def current_platform(self) -> str:
        return self.platform_ctx.current_platform()

    def current_python_environment(self) -> str:
        return self.python_ctx.current_python_environment()

    def current_python_version(self) -> str:
        return self.python_ctx.current_python_version()


if __name__ == "__main__":
    from time import sleep

    context = ChatContext()
    for i in range(3):
        print(f"======= Round {i + 1:02} =======")
        print(f"{context.current_datetime()=}")
        print(f"{context.current_period()=}")
        print(f"{context.current_platform()=}")
        print(f"{context.current_python_environment()=}")
        sleep(1)

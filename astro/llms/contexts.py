from datetime import datetime

from pydantic import BaseModel, Field, computed_field

from astro.utilities.system import (
    PlatformDetails,
    get_platform_details,
    get_platform_str,
)
from astro.utilities.timing import (
    get_datetime_now,
    get_datetime_str,
    get_period_str,
)


class MainChatContext(BaseModel):
    datetime_details: datetime = Field(default_factory=get_datetime_now)
    platform_details: PlatformDetails = Field(default_factory=get_platform_details)

    @computed_field
    def current_datetime(self) -> str:
        return get_datetime_str(self.datetime_details)

    @computed_field
    def current_period(self) -> str:
        return get_period_str(self.datetime_details)

    @computed_field
    def current_platform(self) -> str:
        return get_platform_str(self.platform_details)

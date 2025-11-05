# astro/app/config.py
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from astro.logger import LogLevel, get_loggy

# Load logger
_logger = get_loggy(__file__)


class DisplayTheme(StrEnum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"

    @classmethod
    def from_str(cls, value: str) -> "DisplayTheme":
        match value.lower():
            case "light":
                return DisplayTheme.LIGHT
            case "dark":
                return DisplayTheme.DARK
            case "system":
                return DisplayTheme.SYSTEM
            case _:
                raise ValueError(f"Unknown display theme: {value}")


class StreamlitConfig(BaseModel):
    """Configuration for running the Streamlit app."""

    port: int = Field(
        8501,
        description="Port to run the Streamlit server on.",
        ge=1024,
        le=65535,
        examples=[8501],
    )
    host: str = Field(
        "localhost",
        description="Host address for the Streamlit server (use '0.0.0.0' for all interfaces).",
        examples=["localhost"],
    )
    no_browser: bool = Field(
        False,
        description="If True, do not open the browser automatically on startup.",
    )
    log_level: LogLevel = Field(
        LogLevel.INFO,
        description="Logging level for the Streamlit app.",
    )
    debug_mode: bool = Field(
        False,
        description="Enable Streamlit debug mode (shows extra logs and tracebacks).",
    )
    theme: DisplayTheme = Field(
        DisplayTheme.DARK,
        description="Streamlit theme: 'light', 'dark', or 'system'.",
    )
    run_on_save: bool = Field(
        True,
        description="Automatically rerun the app when source code is saved.",
    )

    model_config = ConfigDict(frozen=True, extra="ignore")


if __name__ == "__main__":
    ...

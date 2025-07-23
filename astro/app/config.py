# astro/app/config.py
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, SecretStr

from astro.logging.base import get_logger, LogLevel
from astro.paths import BASE_ENV_PATH, get_module_dir

# Load logger
_logger = get_logger(__file__)

# Supported APIs for Astro
SUPPORTED_APIS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "TAVILY_API_KEY",
)


class DisplayTheme(StrEnum):
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"

    @classmethod
    def from_str(cls, value: str) -> Self:
        match value.lower():
            case "light":
                return DisplayTheme.LIGHT
            case "dark":
                return DisplayTheme.DARK
            case "system":
                return DisplayTheme.SYSTEM
            case _:
                raise ValueError(f"Unknown display theme: {value}")


class APIKeys(BaseModel):
    # NOTE: Add your API keys you need here

    OPENAI_API_KEY: SecretStr = ""
    ANTHROPIC_API_KEY: SecretStr = ""
    DEEPSEEK_API_KEY: SecretStr = ""
    TAVILY_API_KEY: SecretStr = ""

    model_config = ConfigDict(extra="ignore", frozen=True)

    def __getattribute__(self, name: str) -> str:
        # Intercept attribute access for API key fields
        if name in SUPPORTED_APIS:
            value = super().__getattribute__(name).get_secret_value()
            if not value:
                # Log error and raise if key is not set
                _logger.error(f"API key '{name}' is not set.")
                raise RuntimeError(f"API key '{name}' is not set.")
            return value
        return super().__getattribute__(name)

    def keys_defined(self) -> list[str]:
        """Return a list of API key names that have been set."""
        return [
            field_name
            for field_name, value in self.model_dump().items()
            if isinstance(value, SecretStr) and value
        ]

    def keys_not_defined(self) -> list[str]:
        """Return a list of API key names that have not been set."""
        return [
            field_name
            for field_name, field_value in self.model_dump().items()
            if field_value == "" or field_value is None
        ]


class StreamlitConfig(BaseModel):
    """Configuration for running the Streamlit app."""

    port: int = Field(
        8501,
        description="Port to run the Streamlit server on.",
        ge=1024,
        le=65535,
        example=8501,
    )
    host: str = Field(
        "localhost",
        description="Host address for the Streamlit server (use '0.0.0.0' for all interfaces).",
        example="localhost",
    )
    no_browser: bool = Field(
        False,
        description="If True, do not open the browser automatically on startup.",
    )
    log_level: LogLevel = Field(
        LogLevel.INFO,
        description="Logging level for the Streamlit app.",
        example=LogLevel.INFO,
    )
    debug_mode: bool = Field(
        False,
        description="Enable Streamlit debug mode (shows extra logs and tracebacks).",
    )
    theme: Literal["light", "dark"] = Field(
        "dark",
        description="Streamlit theme: 'light' or 'dark'.",
        pattern="^(light|dark)$",
        example="light",
    )
    run_on_save: bool = Field(
        True,
        description="Automatically rerun the app when source code is saved.",
    )

    model_config = ConfigDict(frozen=True, extra="ignore")


# Global config variables
APIKEYS: APIKeys | None = None


def get_api_keys() -> APIKeys:
    global APIKEYS

    if APIKEYS is not None and isinstance(APIKEYS, APIKeys):
        # Return existing APIKeys object
        _logger.info("Using existing APIKEYS")
        return APIKEYS
    else:
        # Invalid APIKeys object, generate new one
        _logger.info("Loading API keys")
        _logger.debug("Generating new APIKEYS")

    # Application pathing
    _logger.debug("Setting pathing")
    APP_DIR = get_module_dir(__file__)
    LOCAL_ENV_PATH = Path(".env").absolute()
    STREAMLIT_CSS_PATH = APP_DIR / "streamlit.css"

    # Check for streamlit css file
    if not STREAMLIT_CSS_PATH.exists():
        msg = f"Cannot find '{STREAMLIT_CSS_PATH.name}' at '{STREAMLIT_CSS_PATH}'"
        _logger.error(msg, stack_info=True)
        raise FileNotFoundError(msg)

    # Check for .env files
    if not (BASE_ENV_PATH.exists() or LOCAL_ENV_PATH.exists()):
        msg = "No '.env' file found locally or in '~/.astro'"
        _logger.error(
            msg,
            stack_info=True,
            extra={
                "base_env_path": repr(BASE_ENV_PATH),
                "local_env_path": repr(LOCAL_ENV_PATH),
            },
        )
        raise FileNotFoundError(msg)

    # Load values from ~/.astro/.env
    env_data = {}
    if BASE_ENV_PATH.exists():
        _logger.debug(f"Parsing base '.env' file at '{BASE_ENV_PATH}'")
        try:
            base_env = dotenv.dotenv_values(BASE_ENV_PATH)
            _logger.debug(f"Loaded keys from base: {list(base_env.keys())}")
            env_data.update(base_env)
        except Exception as error:
            msg = f"Error occurred while parsing base '.env' file: {error}"
            _logger.error(
                msg, exc_info=True, stack_info=True, extra={"path": str(BASE_ENV_PATH)}
            )
            raise RuntimeError(msg)

    # Load values from .env (local overrides base)
    if LOCAL_ENV_PATH.exists():
        _logger.debug(f"Parsing local .env file at '{LOCAL_ENV_PATH}'")
        try:
            local_env = dotenv.dotenv_values(LOCAL_ENV_PATH)
            _logger.debug(f"Loaded keys from local: {list(local_env.keys())}")
            env_data.update(local_env)
        except Exception as error:
            msg = f"Error occurred while parsing local .env file: {error}"
            _logger.error(
                msg, exc_info=True, stack_info=True, extra={"path": str(LOCAL_ENV_PATH)}
            )
            raise RuntimeError(msg)

    try:
        # Load API keys from .env data safely
        tmp_apikeys = APIKeys(**env_data)
        del env_data  # Remove directly after loading
        _logger.debug("APIKEYS successfully created and frozen.")

        # Report on loaded API keys:
        set_keys = tmp_apikeys.keys_defined()
        not_set = tmp_apikeys.keys_not_defined()

        _logger.info(f"APIKEYS: {len(set_keys)} keys are set.")
        _logger.debug(f"{set_keys=}")
        if len(not_set):
            # Some keys were not loaded
            _logger.warning(f"APIKEYS: {len(not_set)} keys not set.")
            _logger.debug(f"{not_set=}")
    except ValidationError as error:
        # Issue during loading of API keys
        _logger.error(
            f"Config validation error: {error}", exc_info=True, stack_info=True
        )
        raise

    # Safe to set APIKey object and return
    APIKEYS = tmp_apikeys
    return APIKEYS


if __name__ == "__main__":
    print(f"APIKEYS is None? {APIKEYS is None}")
    tmp_apikeys = get_api_keys()
    print(f"APIKEYS is not None? {APIKEYS is not None}")
    print(f"Temp is APIKEYS? {tmp_apikeys is APIKEYS}")
    tmp2_apikeys = get_api_keys()
    print(f"New temp is old temp? {tmp_apikeys is tmp2_apikeys}")
    print(f"New temp is APIKEYS? {tmp2_apikeys is APIKEYS}")
    print(f"Set keys: {', '.join(tmp_apikeys.keys_defined())}")
    print(f"Not set keys: {', '.join(tmp_apikeys.keys_not_defined())}")

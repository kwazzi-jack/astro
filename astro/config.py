# --- Internal Imports ---
import os
from pathlib import Path
from typing import Any

# --- External Imports ---
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, SecretStr, computed_field

# --- Local Imports ---
from astro.logger import get_loggy
from astro.typings import NamedDict

# --- Globals ---
_loggy = get_loggy(__file__)

# API Environment Variables
_API_ENV_NAMES = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "OLLAMA_API_KEY",
    "OLLAMA_BASE_URL",
]

# Single config class instances
_MAIN_CONFIG: "MainConfig | None" = None
_API_CONFIG: "APIConfig | None" = None


# --- Config classes ---
class MainConfig(BaseModel):
    DEBUG_MODE: bool = Field(default=False)


class APIConfig(BaseModel):
    # Set to frozen
    model_config = ConfigDict(frozen=True)

    def _get_env_value(self, env_key: str) -> SecretStr | None:
        if env_key not in os.environ:
            return None
        return SecretStr(os.environ[env_key])

    @computed_field
    @property
    def openai_api_key(self) -> SecretStr | None:
        return self._get_env_value("OPENAI_API_KEY")

    @computed_field
    @property
    def anthropic_api_key(self) -> SecretStr | None:
        return self._get_env_value("ANTHROPIC_API_KEY")

    @computed_field
    @property
    def google_api_key(self) -> SecretStr | None:
        return self._get_env_value("GEMINI_API_KEY")

    @computed_field
    @property
    def deepseek_api_key(self) -> SecretStr | None:
        return self._get_env_value("DEEPSEEK_API_KEY")

    @computed_field
    @property
    def ollama_base_url(self) -> SecretStr | None:
        return self._get_env_value("OLLAMA_BASE_URL") or SecretStr(
            "http://localhost:11434"
        )

    @computed_field
    @property
    def ollama_api_key(self) -> SecretStr | None:
        return self._get_env_value("OLLAMA_API_KEY")

    def openai_set(self) -> bool:
        """Is OpenAI API available to be used?

        Checks if the API key for OpenAI has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.openai_api_key is not None

    def anthropic_set(self) -> bool:
        """Is Anthropic API available to be used?

        Checks if the API key for Anthropic has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.anthropic_api_key is not None

    def google_set(self) -> bool:
        """Is Google (Gemini) API available to be used?

        Checks if the API key for Google (Gemini) has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.google_api_key is not None

    def deepseek_set(self) -> bool:
        """Is DeepSeek API available to be used?

        Checks if the API key for DeepSeek has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.deepseek_api_key is not None

    def ollama_set(self) -> bool:
        """Is Ollama API available to be used?

        Checks if base url for Ollama has been set or
        not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.ollama_base_url is not None

    def is_set_map(self) -> NamedDict:
        return {
            "openai": self.ollama_set(),
            "anthropic": self.anthropic_set(),
            "google": self.google_set(),
            "deepseek": self.deepseek_set(),
            "ollama": self.ollama_set(),
        }

    def is_set(self, provider_or_identifier: str) -> bool:
        if provider_or_identifier.count(":") > 0:
            provider = provider_or_identifier.split(":")[0]
        else:
            provider = provider_or_identifier
        provider_to_api_set = self.is_set_map()
        if provider not in provider_to_api_set:
            raise _loggy.ValueError(f"Provider {provider!r} not supported")
        return provider_to_api_set[provider]

    def get_api(self, provider_or_identifier: str) -> str | None:
        if provider_or_identifier.count(":") > 0:
            provider = provider_or_identifier.split(":")[0]
        else:
            provider = provider_or_identifier
        provider_to_api = {
            "openai": self.openai_api_key,
            "anthropic": self.anthropic_api_key,
            "google": self.google_api_key,
            "deepseek": self.deepseek_api_key,
            "ollama": self.ollama_api_key,
        }
        if provider not in provider_to_api:
            raise _loggy.ValueError(f"Provider {provider!r} not supported")
        result = provider_to_api.get(provider)
        if result is not None:
            return result.get_secret_value()
        return None


# --- Environment Variable Functions ---
def _load_env_file(file_path: Path | None = None) -> NamedDict | None:
    """Load environment variables from a .env file.

    Attempts to load environment variables from the specified .env file path.
    If no path is provided, defaults to '.env' in the current working directory.
    Returns None if the file does not exist.

    Args:
        file_path (Path | None, optional): Optional path to the .env file. Defaults to None, which uses
            the current working directory's .env file.

    Returns:
        NamedDict | None: Dictionary of environment variables if the file exists,
        otherwise None.
    """
    if file_path is None:
        env_path = Path.cwd() / ".env"
    else:
        env_path = file_path

    if not env_path.exists():
        return None

    return dotenv_values(env_path)


def _get_env_dict_from_file(file_path: Path | None = None) -> NamedDict:
    """Get environment variables from a .env file, defaulting to an empty dict.

    Loads environment variables from the specified .env file or returns an empty
    dictionary if the file does not exist or loading fails.

    Args:
        file_path (Path | None, optional): Optional path to the .env file. Defaults to None, which uses
            the current working directory's .env file.

    Returns:
        NamedDict: Dictionary of environment variables, or an empty dict if not found.
    """
    return _load_env_file(file_path) or {}


def _key_intersect_count(dict1: NamedDict, dict2: NamedDict) -> int:
    """Count the number of intersecting keys between two dictionaries.

    Computes the size of the intersection of keys from two NamedDict instances.

    Args:
        dict1 (NamedDict): First dictionary to compare.
        dict2 (NamedDict): Second dictionary to compare.

    Returns:
        int: Number of keys that are present in both dictionaries.
    """
    return len(set(dict1.keys()) & set(dict2.keys()))


# --- Constructor Functions ---
def create_main_config(set_verbose: bool = False) -> MainConfig:
    """Create or return the main configuration instance.
    Initializes a global MainConfig instance if it hasn't been created yet.
    Sets DEBUG_MODE based on the set_verbose parameter or the ASTRO_DEBUG_MODE
    environment variable. If the instance already exists, returns it directly.
    Args:
        set_verbose (bool): If True, enables debug mode. Defaults to False.
    Returns:
        MainConfig: The global main configuration instance.
    """

    global _MAIN_CONFIG

    # Check if initialised and return
    if _MAIN_CONFIG is not None:
        return _MAIN_CONFIG

    # Set environment variables
    main_dict = {"DEBUG_MODE": False}

    # Debug mode
    astro_debug_mode = os.getenv("ASTRO_DEBUG_MODE")
    if set_verbose or (
        astro_debug_mode is not None
        and astro_debug_mode.strip().lower()
        in (
            "1",
            "true",
        )
    ):
        main_dict["DEBUG_MODE"] = True

    # Set global instance and return
    _MAIN_CONFIG = MainConfig(**main_dict)
    return _MAIN_CONFIG


def setup_api_config() -> APIConfig:
    """Create an APIConfig instance by loading environment variables from multiple sources.

    Loads API-related environment variables in order of precedence: defaults,
    secrets file, local .env file, and shell environment. Updates the config
    with higher-precedence sources, logging the process.

    Returns:
        APIConfig: Configured APIConfig instance with loaded environment variables.
    """
    global _API_CONFIG

    # Check if initialised and return
    if _API_CONFIG is not None:
        return _API_CONFIG

    # Lazy import
    from astro.paths import SECRETS_PATH

    # Default environment dictionary
    env_dict: NamedDict = {
        env_api_var_name: None for env_api_var_name in _API_ENV_NAMES
    }
    env_dict["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
    _loggy.debug("Created default environment dictionary", num_keys=len(env_dict))

    # Load from app directory .env file
    if SECRETS_PATH is not None and SECRETS_PATH.exists():
        new_env_dict = _get_env_dict_from_file(SECRETS_PATH)
        _loggy.debug(
            "Found secrets file", num_vars=len(new_env_dict), path=SECRETS_PATH
        )
        env_dict.update(new_env_dict)
    else:
        _loggy.debug("No secrets file set")

    # Load from local .env file
    local_env_path = Path.cwd() / ".env"
    if local_env_path.exists():
        new_env_dict = _get_env_dict_from_file(local_env_path)
        intersect_count = _key_intersect_count(env_dict, new_env_dict)
        _loggy.debug(
            "Found local environment file",
            num_vars=len(new_env_dict),
            overwrite_count=intersect_count,
            path=local_env_path,
        )
        if intersect_count > 0:
            _loggy.debug(
                "Overwriting from environment file",
                overwrite_count={intersect_count},
            )
            env_dict.update(new_env_dict)
        else:
            _loggy.debug("No applicable environment variables to set")
    else:
        _loggy.debug("No local environment file found")

    # Load from bash environment
    new_env_dict = dict(os.environ)
    intersect_count = _key_intersect_count(env_dict, new_env_dict)
    if intersect_count > 0:
        _loggy.debug(
            "Overwriting from shell environment", overwrite_count=intersect_count
        )
        env_dict.update(new_env_dict)
    else:
        _loggy.debug("No applicable shell environment variables to set")

    # Set environment with this dictionary
    for key, value in env_dict.items():
        if value is not None:
            os.environ[key] = value

    # Set global api config instance and return
    _API_CONFIG = APIConfig()
    _loggy.debug(
        "API config created",
        openai_set=_API_CONFIG.openai_set(),
        anthropic_set=_API_CONFIG.anthropic_set(),
        google_set=_API_CONFIG.google_set(),
        deepseek_set=_API_CONFIG.deepseek_set(),
        ollama_set=_API_CONFIG.ollama_set(),
    )

    return _API_CONFIG


def get_api_config() -> APIConfig:
    global _API_CONFIG
    if _API_CONFIG is None:
        _loggy.warning("API Configuration was not set up - Running setup now")
        _API_CONFIG = setup_api_config()
    return _API_CONFIG


# --- Exports ---
__all__ = [
    "APIConfig",
    "get_api_config",
    "setup_api_config",
]

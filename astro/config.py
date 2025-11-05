# --- Internal Imports ---
import os
from pathlib import Path
from typing import Any

# --- External Imports ---
from dotenv import dotenv_values
from pydantic import BaseModel, Field

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
    # OpenAI: gpt-4o, gpt-4o-mini, gpt-5-codex, etc.
    OPENAI_API_KEY: str | None = None

    # Anthropic: claude-4-sonnet, claude-4.5-sonnet, claude-4.5-haiku, etc.
    ANTHROPIC_API_KEY: str | None = None

    # Google: gemini-2.5-flash, gemini-2.0-flash, etc.
    GEMINI_API_KEY: str | None = None

    # DeepSeek AI: deepseek-v3, deepseek-r1, etc.
    DEEPSEEK_API_KEY: str | None = None

    # Ollama (local)
    OLLAMA_API_KEY: str | None = None
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"

    @classmethod
    def from_env_dict(cls, env_dict: dict[str, Any]) -> "APIConfig":
        """Create an APIConfig instance from a dictionary of environment variables.

        Filters the provided environment dictionary to include only the relevant API
        environment variable names and constructs an APIConfig instance with those values.

        Args:
            env_dict: Dictionary containing environment variable names and values.

        Returns:
            APIConfig: An instance of APIConfig with API keys set from the filtered dictionary.
        """
        model_dict = {}
        for key in _API_ENV_NAMES:
            if key in env_dict:
                model_dict[key] = env_dict[key]
        return cls(**model_dict)

    def openai_set(self) -> bool:
        """Is OpenAI API available to be used?

        Checks if the API key for OpenAI has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.OPENAI_API_KEY is not None

    def anthropic_set(self) -> bool:
        """Is Anthropic API available to be used?

        Checks if the API key for Anthropic has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.ANTHROPIC_API_KEY is not None

    def google_set(self) -> bool:
        """Is Google (Gemini) API available to be used?

        Checks if the API key for Google (Gemini) has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.GEMINI_API_KEY is not None

    def deepsek_set(self) -> bool:
        """Is DeepSeek API available to be used?

        Checks if the API key for DeepSeek has been set
        or not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.DEEPSEEK_API_KEY is not None

    def ollama_set(self) -> bool:
        """Is Ollama API available to be used?

        Checks if base url for Ollama has been set or
        not to be used in Astro.

        Returns:
            bool: True if the API key is set, otherwise False.
        """
        return self.OLLAMA_BASE_URL is not None


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

    # Set global api config instance and return
    _API_CONFIG = APIConfig.from_env_dict(env_dict)
    _loggy.debug(
        "API config created",
        openai_set=_API_CONFIG.openai_set(),
        anthropic_set=_API_CONFIG.anthropic_set(),
        google_set=_API_CONFIG.google_set(),
        deepseek_set=_API_CONFIG.deepsek_set(),
        ollama_set=_API_CONFIG.ollama_set(),
    )
    return _API_CONFIG

# --- Internal Imports ---
from pathlib import Path

# --- Globals ---
_loggy = None

# Paths - initialized by setup_paths()
_HOME_DIR: Path | None = None
_ASTRO_DIR: Path | None = None
LOG_DIR: Path | None = None
SECRETS_PATH: Path | None = None
_DATA_DIR: Path | None = None
APPSTATE_PATH: Path | None = None

# Paths setup flag
_PATH_SETUP_DONE = False


def _get_loggy():
    """Get or initialize the global logger for paths module.

    Returns the global logger instance, creating it if it hasn't been initialized yet.
    The logger is configured using the astro.loggings module. Only function like this
    due to circular import issue of logging directory path.

    Returns:
        The logger instance for this module.
    """
    global _loggy

    # If no logger -> set one
    if _loggy is None:
        from astro.logger import Loggy

        _loggy = Loggy(__file__)

    # Return logger, initialized or pre-initialized
    return _loggy


def get_module_dir(file_path: str | None = None) -> Path:
    """Get the directory path of a module.
    Args:
        file_path (str | None, optional): Path to a specific file. If None, returns the parent
            directory of the current module. Defaults to None.
    Returns:
        Path: The resolved directory path. If file is None, returns the parent directory
            of the current module's parent directory. Otherwise, returns the parent
            directory of the specified file.
    Example:
        >>> get_module_dir()
        PosixPath('/home/brian/PhD/astro')
        >>> get_module_dir('/path/to/some/file.py')
        PosixPath('/path/to/some')
    """
    loggy = _get_loggy()
    loggy.debug(f"Getting module directory for file_path: {file_path}")

    # Use astro/paths.py as reference
    if file_path is None:
        result = Path(__file__).parent.parent.resolve()
        loggy.debug(f"Using current module reference, returning: {result}")
        return result

    # Use specified astro file
    else:
        try:
            result = Path(file_path).parent.resolve()
            loggy.debug(f"Using specified file path, returning: {result}")
            return result
        except Exception as error:
            raise loggy.IOError(
                f"Error resolving path for file: {file_path}",
                caused_by=error,
                file_path=file_path,
            )


def get_file_modification_time(file_path: Path) -> float:
    """Get the modification time of a file as a timestamp.

    Args:
        file_path: Path to the file.

    Returns:
        float: Modification time as timestamp.

    Raises:
        OSError: If the file cannot be accessed.
    """
    loggy = _get_loggy()
    loggy.debug(f"Getting modification time for: {file_path}")

    try:
        mod_time = file_path.stat().st_mtime
        loggy.debug(f"File {file_path} modified at: {mod_time}")
        return mod_time
    except Exception as error:
        raise loggy.OSError(
            f"Cannot access file modification time: {file_path}",
            file_path=file_path,
            caused_by=error,
        )


def setup_paths():
    """Initialize all path constants and create necessary directories.

    This function must be called before using any path-dependent functionality.
    Creates the directory structure if it doesn't exist.
    """
    global \
        _HOME_DIR, \
        _ASTRO_DIR, \
        LOG_DIR, \
        SECRETS_PATH, \
        _STATE_DIR, \
        _STORES_DIR, \
        _DATA_DIR, \
        REPOSITORY_DIR, \
        APPSTATE_PATH, \
        _PATH_SETUP_DONE

    if _PATH_SETUP_DONE:
        return

    try:
        # Astro's home
        _HOME_DIR = Path.home()
        _ASTRO_DIR = _HOME_DIR / ".astro"
        _ASTRO_DIR.mkdir(exist_ok=True)

        # Path for logs
        LOG_DIR = _ASTRO_DIR / "logs"
        LOG_DIR.mkdir(exist_ok=True)

        # Path for configs
        SECRETS_PATH = _ASTRO_DIR / ".secrets"

        # State and store directories
        _STATE_DIR = _ASTRO_DIR / "state"
        _STATE_DIR.mkdir(exist_ok=True)

        # Data and repository directory
        _DATA_DIR = _ASTRO_DIR / "data"
        _DATA_DIR.mkdir(exist_ok=True)

        # Path to application state
        APPSTATE_PATH = _STATE_DIR / "appstate"

    # Something went wrong when setting up paths
    except Exception as error:
        raise IOError("Error occurred while setting up paths") from error

    # Flag that paths are setup
    _PATH_SETUP_DONE = True


def get_state_dir() -> Path:
    """Get the state directory path where runtime state is kept.

    Returns:
        Path: The state directory path

    Raises:
        RuntimeError: If paths have not been setup yet
    """
    if not _PATH_SETUP_DONE or _STATE_DIR is None:
        raise RuntimeError("Paths not initialized. Call setup_paths() first.")
    return _STATE_DIR


# Run when loading module
setup_paths()

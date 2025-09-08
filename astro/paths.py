import json
import shutil
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from pathlib import Path
from typing import Any, Generic, TypeAlias

from astro.typings import (
    PathDict,
    RecordableModel,
    RecordableModelType,
    _options_to_str,
    _path_dict_to_str_dict,
    _str_dict_to_path_dict,
    type_name,
)

# Global logger variable
_logger = None


def _get_logger():
    """Get or initialize the global logger for paths module.

    Returns the global logger instance, creating it if it hasn't been initialized yet.
    The logger is configured using the astro.loggings module. Only function like this
    due to circular import issue of logging directory path.

    Returns:
        The logger instance for this module.
    """
    global _logger

    # If no logger -> set one
    if _logger is None:
        from astro.loggings.base import Loggy

        _logger = Loggy(__file__)

    # Return logger, initialized or pre-initialized
    return _logger


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
    logger = _get_logger()
    logger.debug(f"Getting module directory for file_path: {file_path}")

    # Use astro/paths.py as reference
    if file_path is None:
        result = Path(__file__).parent.parent.resolve()
        logger.debug(f"Using current module reference, returning: {result}")
        return result

    # Use specified astro file
    else:
        result = Path(file_path).parent.resolve()
        logger.debug(f"Using specified file path, returning: {result}")
        return result


def get_file_modification_time(file_path: Path) -> float:
    """Get the modification time of a file as a timestamp.

    Args:
        file_path: Path to the file.

    Returns:
        float: Modification time as timestamp.

    Raises:
        OSError: If the file cannot be accessed.
    """
    logger = _get_logger()
    logger.debug(f"Getting modification time for: {file_path}")

    try:
        mod_time = file_path.stat().st_mtime
        logger.debug(f"File {file_path} modified at: {mod_time}")
        return mod_time
    except Exception as error:
        logger.error(f"Failed to get modification time for {file_path}: {error}")
        raise OSError(f"Cannot access file modification time: {file_path}") from error


def find_latest_log_file(log_dir: Path, pattern: str = "*.jsonl") -> Path | None:
    """Find the most recent log file matching the pattern in a directory.

    Args:
        log_dir: Directory to search for log files.
        pattern: Glob pattern to match files (default: "*.jsonl").

    Returns:
        Path to the most recent log file, or None if no files found.
    """
    logger = _get_logger()
    logger.debug(f"Finding latest log file in {log_dir} with pattern: {pattern}")

    if not log_dir.exists() or not log_dir.is_dir():
        logger.warning(f"Log directory does not exist or is not a directory: {log_dir}")
        return None

    try:
        log_files = list(log_dir.glob(pattern))
        if not log_files:
            logger.info(f"No log files found in {log_dir} matching {pattern}")
            return None

        # Find the file with the latest modification time
        latest_file = max(log_files, key=get_file_modification_time)
        logger.info(f"Latest log file found: {latest_file}")
        return latest_file

    except Exception as error:
        logger.error(f"Error finding latest log file in {log_dir}: {error}")
        return None


def get_available_log_files(log_dir: Path, pattern: str = "*.jsonl") -> list[Path]:
    """Get all available log files, sorted by modification time (newest first).

    Args:
        log_dir: Directory to search for log files.
        pattern: Glob pattern to match files (default: "*.jsonl").

    Returns:
        List of log file paths sorted by modification time, newest first.
    """
    logger = _get_logger()
    logger.debug(f"Getting available log files in {log_dir} with pattern: {pattern}")

    if not log_dir.exists() or not log_dir.is_dir():
        logger.warning(f"Log directory does not exist or is not a directory: {log_dir}")
        return []

    try:
        log_files = list(log_dir.glob(pattern))
        if not log_files:
            logger.info(f"No log files found in {log_dir} matching {pattern}")
            return []

        # Sort by modification time, newest first
        sorted_files = sorted(log_files, key=get_file_modification_time, reverse=True)
        logger.info(f"Found {len(sorted_files)} log files in {log_dir}")
        return sorted_files

    except Exception as error:
        logger.error(f"Error getting available log files in {log_dir}: {error}")
        return []


def read_markdown_file(markdown_file_path: str | Path) -> str:
    """Read and return the contents of a markdown file.
    Args:
        markdown_file_path (str | Path): Path to the markdown file to read.
            Can be either a string path or a Path object.
    Returns:
        str: The contents of the markdown file with leading and trailing
            whitespace stripped.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        RuntimeError: If there are issues reading the file (permissions, etc.).
    """
    logger = _get_logger()
    logger.debug(f"Reading markdown file: {markdown_file_path}")

    # If string file path, convert to Path
    if isinstance(markdown_file_path, str):
        markdown_file_path = Path(markdown_file_path)
        logger.debug(f"Converted string path to Path object: {markdown_file_path}")

    # Validate file exists
    if not markdown_file_path.exists():
        logger.error(f"Markdown file does not exist: {markdown_file_path}")
        raise FileNotFoundError(f"File `{markdown_file_path}` does not exist")

    # Return contents of markdown file
    try:
        with open(markdown_file_path, encoding="utf-8") as file:
            contents = file.read().strip()
            logger.info(
                f"Successfully read markdown file: {markdown_file_path} ({len(contents)} characters)"
            )
            return contents
    # Error while opening file
    except Exception as error:
        logger.error(f"Failed to read markdown file {markdown_file_path}: {error}")
        raise OSError(
            f"Error while trying to open markdown file '{markdown_file_path}'"
        ) from error


class ModelFileStore(Generic[RecordableModelType]):
    """Index and manage file paths for traceable model objects in a directory.

    Provides a persistent mapping between unique model identifiers and file paths
    within a specified root directory. Supports adding, removing, and retrieving
    model objects, as well as maintaining an index file for fast lookup and recovery.
    Ensures type safety for all operations and validates model types at runtime.

    Attributes:
        `name` (`str`): Name of the root directory used for indexing.
        `root_dir` (`Path`): Root directory where model files and index are stored.
        `model_type` (`type(TraceableModel)`): Type of model managed by this indexer (must be `TraceableModel`).
        `index_file` (`Path`): Path to the index file storing the mapping.

    Args:
        `root_dir` (`Path`): Directory where model files and the index file are stored.
        `model_type` (`type(TraceableModel)`): Type of model to be indexed. Must be a subclass of `TraceableModel`.
    """

    def __init__(self, root_dir: Path, model_type: type[RecordableModelType]):
        """Initialize the `ModelFileStore` with a root directory and model type.

        Creates the root directory if it does not exist, validates the model type,
        and loads or initializes the index file.

        Args:
            `root_dir`: Directory where model files and the index file are stored.
            `model_type`: Type of model to be indexed. Must be a subclass of `TraceableModel`.

        Raises:
            `ValueError`: If model_type is not a subclass of `TraceableModel`.
            `IOError`: If `ModelFileStore` encounters an error while saving or loading the index file.
        """
        logger = _get_logger()
        logger.debug(
            f"Initializing ModelFileStore for {model_type.__name__ if hasattr(model_type, '__name__') else model_type} at {root_dir}"
        )

        # Create assigned directory if it does not exist
        root_dir.mkdir(exist_ok=True)
        logger.debug(f"Created/verified directory: {root_dir}")

        # Validate `model_type` is subclass of `TraceableModel`
        if not (
            isinstance(model_type, type) and issubclass(model_type, RecordableModel)
        ):
            error = ExpectedVarType(
                var_name="model_type",
                got=model_type,
                expected=RecordableModel,
            )
            logger.error(error.to_log())
            raise error

        # Attributes
        self._name = root_dir.name
        self._root_dir = root_dir
        self._model_type = model_type
        self._index_file = root_dir / "index"
        self._index_map = {}  # type: PathDict

        logger.debug(f"Set up ModelFileStore attributes for {self._name}")

        # Index file present -> load from file
        if self._index_file.exists():
            logger.debug(f"Loading existing index file: {self._index_file}")
            self._index_map = self._load_index()
            logger.info(
                f"Loaded {len(self._index_map)} entries from existing index for {self._name}"
            )

        # No index file -> create clean file
        else:
            logger.debug(f"Creating new index file: {self._index_file}")
            self._index_map = {}
            self._save_index()
            logger.info(f"Created new empty index for {self._name}")

    @property
    def name(self) -> str:
        """Name of the root directory used for indexing."""
        return self._name

    @property
    def root_dir(self) -> Path:
        """Root directory where model files and index are stored."""
        return self._root_dir

    @property
    def model_type(self) -> type[RecordableModelType]:
        """Type of model managed by this indexer (must be `TraceableModel`)."""
        return self._model_type

    @property
    def index_file(self) -> Path:
        """Path to the index file storing the mapping."""
        return self._index_file

    def _save_index(self):
        """Save the current index mapping to the index file.

        Serializes the internal path dictionary to a JSON file for persistence.

        Raises:
            `IOError`: If an error occurs while writing to the index file.
        """

        # Convert path dictionary to strings (serialiazable)
        str_dict = _path_dict_to_str_dict(self._index_map)

        try:
            # Open index file and dumpy
            with open(self.index_file, "w") as file:
                json.dump(str_dict, file)

        except Exception as error:
            # An error occurred - propagate up
            raise IOError(
                f"Error occurred while loading from index file `{self.index_file}`"
            ) from error

    def _load_index(self) -> PathDict:
        """Load the index mapping from the index file.

        Reads the index file and reconstructs the internal path dictionary.

        Returns:
            `PathDict`: Dictionary mapping UIDs to file paths.

        Raises:
            `IOError`: If an error occurs while reading or parsing the index file.
        """

        try:
            # Load string dictionary from index file
            with open(self.index_file, "r") as file:
                str_dict = json.load(file)

            # Convert string dictionary to path dictionary
            path_dict = _str_dict_to_path_dict(str_dict)

            # Remove entry if no file
            return {key: value for key, value in path_dict.items() if value.exists()}

        except Exception as error:
            # An error occurred - propagate up
            raise IOError(
                f"Error occurred while saving to index file `{self.index_file}`"
            ) from error

    def _save_object(self, obj: RecordableModelType):
        """Serialize and save a model object to its associated file path.

        Validates the object type and writes its JSON representation to disk.

        Args:
            `obj`: Model object to save. Must be an instance of the configured `Model` type.

        Raises:
            `ValueError`: If `obj` is not an instance of the configured model type.
            `IOError`: If an error occurs while saving the object to disk.
        """

        # Input type validation
        if not isinstance(obj, self.model_type):
            error = ExpectedVarType(
                var_name="obj",
                got=type(obj),
                expected=self.model_type,
            )
            logger.error(error.to_log())
            raise error

        try:
            # Fetch file path based on UID
            file_path = self[obj.uid]

            # Dump as json and save (requires as `TraceableModel`)
            with open(file_path, "w") as file:
                json.dump(obj.model_dump(mode="json"), file)

        except Exception as error:
            # An error occurred - propagate up
            raise IOError(
                f"Error while saving `{type_name(obj)}` ({obj.uid})"
            ) from error

    def _load_object(self, key: str) -> RecordableModelType:
        """Load and deserialize a model object from its associated file path.

        Retrieves the file path for the given key and reconstructs the model object.

        Args:
            ``key`: Unique identifier for the `Model` object.

        Returns:
            `Model`: The loaded and validated `Model` object.

        Raises:
            `ValueError`: If `key` is not a string.
            `IOError`: If an error occurs while loading or parsing the file.
        """

        # Input type validation
        if not isinstance(key, str):
            error = KeyStrError(got=type(key))
            logger.error(error.to_log())
            raise error

        file_path = None
        try:
            # Fetch file path based on key
            file_path = self[key]

            # Load object dictionary from file
            with open(file_path, "r") as file:
                contents = json.load(file)

            # Create assigned model type based on contents
            return self.model_type.model_validate(contents)

        except Exception as error:
            # An error occurred - propagate up
            outer_error = LoadError(path_or_uid=file_path)
            logger.error(**outer_error.to_log())
            raise outer_error from error

    def __getitem__(self, key: str) -> Path:
        """Retrieve the file path associated with a given key.

        Fetches the Path object mapped to the specified string key from the index.

        Args:
            `key` (`str`): Unique identifier for the model object.

        Returns:
            `Path`: The file path associated with the key.

        Raises:
            `ValueError`: If `key` is not a string.
            `KeyError`: If `key` is not present in the index.
        """

        # Input type validation
        if not isinstance(key, str):
            error = KeyStrError(got=type(key))
            logger.error(**error.to_log())
            raise error

        # No entry associated with key
        if key not in self:
            error = NoEntryKeyError(key_value=key)
            logger.error(**error.to_log())
            raise error

        # Return path associated with key
        return self._index_map[key]

    def __setitem__(self, key: str, value: Path):
        """Set the file path associated with a given key.

        Assigns the specified Path value to the string key in the index and persists the change.

        Args:
            `key` (`str`): Unique identifier for the model object.
            `value` (`Path`): File path to associate with the key.

        Raises:
            `ValueError`: If `key` is not a string or `value` is not a Path.
            `IOError`: If an error occurs while updating the index file.
        """

        # Input type validation
        if not isinstance(key, str):
            error = KeyStrError(got=type(key))
            logger.error(**error.to_log())
            raise error

        if not isinstance(value, Path):
            error = ValueTypeError(got=type(value), expected=Path)
            logger.error(**error.to_log())
            raise error

        # Set associated key with given value
        self._index_map[key] = value

        # Update index file
        self._save_index()

    def __delitem__(self, key: str):
        """Delete the file path associated with a given key.

        Removes the Path object mapped to the specified string key from the index,
        deletes the associated file from the file system, and persists the change
        by updating the index file.

        Args:
            `key` (`str`): Unique identifier for the model object.

        Raises:
            `ValueError`: If `key` is not a string.
            `KeyError`: If `key` is not present in the index.
            `IOError`: If an error occurs while deleting the file or updating the index file.
        """
        # Input type validation
        if not isinstance(key, str):
            error = KeyStrError(got=type(key))
            logger.error(**error.to_log())
            raise error

        # No entry associated with key
        if key not in self:
            error = NoEntryKeyError(key_value=key)
            logger.error(**error.to_log())
            raise error

        # Fetch file path based on key and delete
        file_path = self[key]
        file_path.unlink(missing_ok=True)

        # Remove entry in index
        del self._index_map[key]

        # Update index file
        self._save_index()

    def __iter__(self) -> Iterator[str]:
        """Iterate over the keys in the index.

        Yields:
            str: Each key in the index.
        """
        return iter(self._index_map)

    def __contains__(self, key_or_obj: str | RecordableModelType) -> bool:
        """Check if a key or model object is present in the index.

        Determines whether the given string key or model object's UID exists
        in the internal index mapping. For model objects, checks the UID attribute.

        Args:
            `key_or_obj` (`str | Model`): String key or model object to check for presence.

        Returns:
            True if the key or object's UID is in the index, False otherwise.

        Raises:
            `ValueError`: If `key_or_obj` is not a string or an instance of the configured model type.
        """
        # Input type validation
        if not isinstance(key_or_obj, (str, self.model_type)):
            error = ExpectedVarType(
                var_name="key_or_obj",
                got=type(key_or_obj),
                expected=(str, self.model_type),
            )
            logger.error(**error.to_log())
            raise error

        # Return if key or UID has associated entry
        if isinstance(key_or_obj, str):
            return key_or_obj in self._index_map
        else:
            return key_or_obj.uid in self._index_map

    def get_model(self, key: str, default: Any = None) -> RecordableModelType | Any:
        """Retrieve an object by key, with a default value if not found.

        Args:
            `key` (`str`): The unique identifier for the model object.
            `default` (`Any`, optional): The value to return if the key is not found.
                Defaults to `None`.

        Returns:
            `Model` | `Any`: The loaded model object if the key is found,
            otherwise the default value.
        """
        # If key present, load object
        if key in self:
            try:
                return self._load_object(key)

            # Return default if any error
            except LoadError:
                logger.debug("Encountered load error. Returning default")
                return default

        # Key not found, return default
        return default

    def add_model(self, obj: RecordableModelType):
        """Add a model object to the index.

        Generates a file path for the object based on its UID, stores the
        mapping in the index, and saves the object to disk.

        Args:
            `obj` (`Model`): Model object to add to the index.

        Raises:
            `ValueError`: If `obj` is not an instance of the configured model type.
            `IOError`: If an error occurs while saving the object or updating the index.
        """
        # Input type validation
        if not isinstance(obj, self.model_type):
            raise ExpectedVarType(
                var_name="obj", got=type(obj), expected=self.model_type
            )

        # Create file path based on input object
        file_path = self.root_dir / obj.uid

        # Set object based on UID and file path
        self.__setitem__(obj.uid, file_path)

        # Update index file
        self._save_object(obj)
        self._save_index()

    def remove_model(self, key_or_obj: str | RecordableModelType):
        """Remove a model object from the index.

        Deletes the file associated with the object (or key), removes the
        mapping from the index, and updates the index file.

        Args:
            `key_or_obj` (`str | Model`): Key or model object to remove from the index.

        Raises:
            `ValueError`: If `key_or_obj` is not a string or an instance of the configured model type.
            `IOError`: If an error occurs while deleting the file or updating the index.
        """

        # Input type validation
        if not isinstance(key_or_obj, (str, self.model_type)):
            raise ExpectedVarType(
                var_name="key_or_obj",
                got=type(key_or_obj),
                expected=(str, self.model_type),
            )

        # Delete entry associated with key or UID
        if isinstance(key_or_obj, str):
            del self[key_or_obj]
        else:
            del self[key_or_obj.uid]

    def clear(self):
        """Clear all entries from the index and delete associated files.

        Removes all file mappings from the index, deletes the corresponding
        files from the file system, and updates the index file.
        """
        # For each entry -> delete associated file
        for path in self._index_map.values():
            path.unlink(missing_ok=True)

        # Clear all local entries
        self._index_map = {}

        # Update index file
        self._save_index()

    def __len__(self) -> int:
        """Returns the number of items in the path index.

        Returns:
            `int`: The length of the internal index map.
        """
        return len(self._index_map)

    def keys(self) -> KeysView[str]:
        """Return a view of the keys in the index.

        Returns:
            `KeysView[str]`: A view object that displays a list of all keys in the index.
        """
        return self._index_map.keys()

    def values(self) -> ValuesView[Path]:
        """Return a view of the values in the index.

        Returns:
            `ValuesView[Path]`: A view object that displays a list of all values in the index.
        """
        return self._index_map.values()

    def items(self) -> ItemsView[str, Path]:
        """Return a view of the items in the index.

        Returns:
            `ItemsView[str, Path]`: A view object that displays a list of all (key, value) pairs in the index.
        """
        return self._index_map.items()

    def __str__(self) -> str:
        """Return a user-friendly string representation of the ModelFileStore.

        Returns:
            `str`: A string describing the indexer, including the model type and root directory.
        """
        return f"ModelFileStore for {self.model_type.__name__} at {self.root_dir}"

    def __repr__(self) -> str:
        """Return an unambiguous string representation of the ModelFileStore.

        Returns:
            `str`: A string that could be used to recreate the ModelFileStore instance.
        """
        return f"ModelFileStore(root_dir={self.root_dir!r}, model_type={self.model_type!r})"


# Global path variables - initialized by setup_paths()
_HOME_DIR: Path | None = None
_ASTRO_DIR: Path | None = None
_INSTALL_HOME_DIR: Path | None = None

_LOG_DIR: Path | None = None
_BASE_SECRETS_PATH: Path | None = None
_STATE_DIR: Path | None = None
_STORES_DIR: Path | None = None

# Paths setup flag
_PATH_SETUP_DONE = False


def setup_paths():
    """Initialize all path constants and create necessary directories.

    This function must be called before using any path-dependent functionality.
    Creates the directory structure if it doesn't exist.
    """
    global \
        _HOME_DIR, \
        _ASTRO_DIR, \
        _LOG_DIR, \
        _BASE_SECRETS_PATH, \
        _STATE_DIR, \
        _STORES_DIR, \
        _PATH_SETUP_DONE

    if _PATH_SETUP_DONE:
        return

    try:
        # Astro's home
        _HOME_DIR = Path.home()
        _ASTRO_DIR = _HOME_DIR / ".astro"
        _ASTRO_DIR.mkdir(exist_ok=True)

        # Path for logs
        _LOG_DIR = _ASTRO_DIR / "logs"
        _LOG_DIR.mkdir(exist_ok=True)

        # Path for configs
        _BASE_SECRETS_PATH = _ASTRO_DIR / ".secrets"

        # State and store directories
        _STATE_DIR = _ASTRO_DIR / "state"
        _STATE_DIR.mkdir(exist_ok=True)

        _STORES_DIR = _STATE_DIR / "stores"
        _STORES_DIR.mkdir(exist_ok=True)

    # Something went wrong when setting up paths
    except Exception as error:
        raise IOError("Error occurred while setting up paths") from error

    # Flag that paths are setup
    _PATH_SETUP_DONE = True


# Useful type aliases
ModelTypeDict: TypeAlias = dict[str, type[RecordableModel]]
StoreDict: TypeAlias = dict[str, ModelFileStore[RecordableModelType]]

# Main store white list and mapping
_STORE_WHITE_LIST_MAP: ModelTypeDict = {}
_STORE_MAP: StoreDict = {}


def _load_store_white_list():
    """Load the white list of traceable models allowed for storage.

    Imports the necessary model classes and initializes the global _STORE_WHITE_LIST
    if it has not been set yet. This controls which model types can be persisted
    in the file stores. The list is only created once to avoid re-initialization.

    Note:
        Restart the application if changes to the white list are needed, as it
        is not re-created if already populated.
    """
    logger = _get_logger()
    logger.debug("Loading store white list")

    global _STORE_WHITE_LIST_MAP

    # Non-empty list will not get re-created
    # IMPORTANT - B - restart Astro instead
    if len(_STORE_WHITE_LIST_MAP):
        logger.debug(
            f"Store white list already loaded with {len(_STORE_WHITE_LIST_MAP)} entries"
        )
        return

    try:
        # List of TraceableModels to white list
        # IMPORTANT - B - This governs which models can be stored
        from astro.llms import LLMConfig

        # Create list based on above
        _STORE_WHITE_LIST_MAP = {LLMConfig.__name__: LLMConfig}
        logger.info(
            f"Store white list loaded successfully: {list(_STORE_WHITE_LIST_MAP.keys())}"
        )

    except ImportError as error:
        logger.error(f"Failed to import required models for white list: {error}")
        raise
    except Exception as error:
        logger.error(f"Unexpected error loading store white list: {error}")
        raise


def _is_store_dir(dir_path: Path) -> bool:
    """Check if a directory is a valid store directory.

    Validates that the given path is a directory, exists, is not empty, and contains
    an 'index' file, indicating it is a store directory.

    Args:
        `dir_path` (`Path`): The directory path to check.

    Returns:
        `bool`: True if the directory is a valid store directory, False otherwise.

    Raises:
        `ValueError`: If `dir_path` is not a `Path` instance.
    """
    logger = _get_logger()
    logger.debug(f"Checking if directory is valid store: {dir_path}")

    # Input type validation
    if not isinstance(dir_path, Path):
        error = ExpectedVarType(var_name="dir_path", got=type(dir_path), expected=Path)
        logger.error(**error.to_log())
        raise error

    # If not in white-list -> False
    if dir_path.name not in _STORE_WHITE_LIST_MAP:
        logger.debug(f"Directory {dir_path.name} not in whitelist")
        return False

    # If directory doesn't exist or not a directory -> False
    if not dir_path.exists() or not dir_path.is_dir():
        logger.debug(f"Directory {dir_path} does not exist or is not a directory")
        return False

    # Directory contents
    dir_contents = list(dir_path.iterdir())

    # If empty directory -> False
    if len(dir_contents) == 0:
        logger.debug(f"Directory {dir_path} is empty")
        return False

    # If no index file -> False
    if not any(item.name == "index" for item in dir_contents):
        logger.debug(f"Directory {dir_path} does not contain index file")
        return False

    # Assume true
    logger.debug(f"Directory {dir_path} is a valid store directory")
    return True


def _setup_store_map():
    logger = _get_logger()
    logger.debug("Setting up store map")

    global _STORE_MAP

    # Issue with missing path
    if not _PATH_SETUP_DONE or _STORES_DIR is None:
        error = SetupError(
            cause="Paths not initialized",
            path=_STORES_DIR,
            path_setup_flag=_PATH_SETUP_DONE,
        )
        logger.error(**error.to_log())
        raise error

    # Only setup once
    if len(_STORE_MAP) != 0:
        logger.debug(f"Store map already setup with {len(_STORE_MAP)} stores")
        return

    logger.debug(f"Scanning stores directory: {_STORES_DIR}")
    store_path: Path | None = None
    try:
        # Get list of stores
        for store_path in _STORES_DIR.iterdir():
            logger.debug(f"Processing store path: {store_path}")

            # Is it a store directory -> setup
            if _is_store_dir(store_path):
                # Get model type, name and then store
                model_name = store_path.name
                model_type = _STORE_WHITE_LIST_MAP[model_name]
                logger.debug(f"Setting up store for model type: {model_name}")

                model_store = ModelFileStore[model_type](
                    root_dir=store_path, model_type=model_type
                )

                # Add model store to map
                _STORE_MAP[model_store.name] = model_store
                logger.info(f"Successfully loaded existing store for {model_name}")

            # If not a store, remove
            # Is directory -> delete contents and remove
            elif store_path.is_dir():
                logger.warning(f"Removing invalid store directory: {store_path}")
                shutil.rmtree(store_path)
                store_path.rmdir()

            # Is file -> delete
            else:
                logger.warning(f"Removing invalid store file: {store_path}")
                store_path.unlink()
    except Exception as error:
        outer_error = SetupError(
            cause="Error occurred while setting up store map",
            current_path=store_path,
            path=_STORES_DIR,
        )
        logger.error(**outer_error.to_log(), exc_info=True)
        raise outer_error from error

    # Ensure all white-listed model types have a store
    logger.debug("Ensuring all whitelisted model types have stores")
    model_name: str | None = None
    model_type: type[RecordableModel] | None = None
    try:
        for model_name, model_type in _STORE_WHITE_LIST_MAP.items():
            if model_name not in _STORE_MAP:
                logger.debug(f"Creating new store for model type: {model_name}")
                store_path = _STORES_DIR / model_name
                model_store = ModelFileStore[model_type](
                    root_dir=store_path, model_type=model_type
                )
                _STORE_MAP[model_store.name] = model_store
                logger.info(f"Successfully created new store for {model_name}")
    except Exception as error:
        outer_error = SetupError(
            cause=f"Error occurred while creating new store for {model_name} ({model_type})",
        )
        logger.error(**outer_error.to_log(), exc_info=True)
        raise outer_error from error

    logger.info(
        f"Store map setup completed with {len(_STORE_MAP)} stores: {list(_STORE_MAP.keys())}"
    )


def setup_store():
    """Set up the model file store system.

    This is separate from path setup and handles TraceableModel serialization.

    Raises:
        RuntimeError: If paths have not been initialized via setup_paths().
    """
    logger = _get_logger()
    logger.debug("Starting store setup")

    # Check if paths are initialized
    if _ASTRO_DIR is None:
        error = SetupError(
            cause="Paths not initialized",
            path=_ASTRO_DIR,
            path_setup_flag=_PATH_SETUP_DONE,
        )
        logger.warning("Paths not initialized. Call astro.paths.setup_paths() first.")
        logger.error(**error.to_log())
        raise error

    # Load white list
    logger.debug("Loading store white list")
    _load_store_white_list()

    # Load store map
    # CAUTION - B - Deletes non-white listed model stores
    logger.debug("Setting up store map")
    _setup_store_map()

    logger.info(f"Store setup completed. Available stores: {list(_STORE_MAP.keys())}")


def get_model_file_store(model_type: type[RecordableModel]) -> ModelFileStore:
    """Create and return a ModelFileStore instance for the specified model type.

    Validates the input model type, constructs a directory name based on the model
    type's name, and initializes a ModelFileStore in the corresponding objects subdirectory.

    Args:
        `model_type` (`type[TraceableModel]`): The model type to create a store for.
            Must be a subclass of `TraceableModel`.

    Returns:
        `ModelFileStore[model_type]`: A configured ModelFileStore instance for the given model type.

    Raises:
        `ValueError`: If `model_type` is not a subclass of `TraceableModel`.
    """
    logger = _get_logger()
    logger.debug(
        f"Getting model file store for type: {model_type.__name__ if hasattr(model_type, '__name__') else model_type}"
    )

    # Input type validation
    if not isinstance(model_type, type) or not issubclass(model_type, RecordableModel):
        error = ExpectedVarType(
            var_name="model_type",
            got=model_type,
            expected=RecordableModel,
        )
        logger.error(**error.to_log())
        raise error

    # Extract name
    model_name = model_type.__name__
    logger.debug(f"Looking for store for model: {model_name}")

    # Entry not found in store map
    # Brian - Might be redundant double check
    if model_name not in _STORE_WHITE_LIST_MAP or model_name not in _STORE_MAP:
        error = NoEntryKeyError(
            key_value=model_name, sources=("whitelist", "store map")
        )
        logger.error(**error.to_log())
        raise error

    # Return corresponding model file store
    logger.debug(f"Successfully retrieved store for model: {model_name}")
    return _STORE_MAP[model_name]


def save_model_file_to_store(model_file: RecordableModel):
    logger = _get_logger()
    logger.debug(
        f"Attempting to save model {model_file.uid} of type {type(model_file).__name__}"
    )

    # Extract input type
    model_type = type(model_file)

    # Input type validation
    if not isinstance(model_file, RecordableModel):
        error = ExpectedVarType(
            var_name="model_file", got=type(model_file), expected=RecordableModel
        )
        logger.error(**error.to_log())
        raise error

    # Not a valid model file to store
    if model_type.__name__ not in _STORE_WHITE_LIST_MAP:
        error = NoEntryKeyError(key_value=model_type.__name__, sources="whitelist")
        logger.error(**error.to_log())
        raise error

    try:
        # Get model file store
        model_store = get_model_file_store(model_type)

        # Add new model to store
        model_store.add_model(model_file)
        logger.info(
            f"Successfully saved model {model_file.uid} to {model_type.__name__} store"
        )

    # Error occurred while trying to load file store and save model
    except Exception as error:
        outer_error = SaveError(
            path_or_uid=model_file.uid,
            obj_to_save=model_file,
            save_to="model file store",
        )
        logger.error(**outer_error.to_log(), exc_info=True)
        raise outer_error from error


def load_model_file_from_store(key: str) -> RecordableModel:
    logger = _get_logger()
    logger.debug(f"Attempting to load model with key: {key}")

    # Input type validation
    if not isinstance(key, str):
        error = KeyStrError(got=type(key))
        logger.error(**error.to_log())

    # Check stores for entry
    matches: list[ModelFileStore] = []
    for store in _STORE_MAP.values():
        # If entry in store -> add match
        if key in store:
            matches.append(store)

    # No matches
    if len(matches) == 0:
        error = NoEntryKeyError(key_value=key, sources="model file stores")

    # Contains duplicates -> UID matching problem
    if len(matches) > 1:
        matches_str = _options_to_str([store.name for store in matches])
        logger.error(f"Duplicate UID {key} found in stores: {matches_str}")
        raise ValueError(
            f"Duplicate UID `{key}` found in multiple stores: {matches_str}"
        )

    # Get store
    model_store = matches[0]
    logger.debug(f"Found model {key} in store: {model_store.name}")

    try:
        # Return model from model store
        model = model_store.get_model(key)
        logger.info(f"Successfully loaded model {key} from {model_store.name} store")
        return model
    except Exception as error:
        logger.error(
            f"Failed to load model {key} from store {model_store.name}: {error}"
        )
        raise RuntimeError(
            f"Error occurred while loading model {key} from store `{model_store.name}`"
        ) from error


def remove_model_file_from_store(*keys: str):
    """Remove model files from their respective stores by UID.

    Args:
        *keys: One or more string UIDs of models to remove from stores.

    Raises:
        ValueError: If any key is not a string.
        KeyError: If any key is not found in any store.
        RuntimeError: If an error occurs while removing models from stores.
    """
    logger = _get_logger()
    logger.debug(f"Attempting to remove models with keys: {keys}")

    # Input type validation for all keys
    for key in keys:
        if not isinstance(key, str):
            logger.error(f"Invalid key type: {type(key)} for key: {key}")
            raise _expected_key_str_value_error(got=type(key))

    # Process each key
    for key in keys:
        logger.debug(f"Processing removal of key: {key}")

        # Check stores for entry
        matches: list[ModelFileStore] = []
        for store in _STORE_MAP.values():
            # If entry in store -> add match
            if key in store:
                matches.append(store)

        # No matches
        if len(matches) == 0:
            logger.warning(f"Model with key {key} not found for removal")
            raise _no_entry_key_error(key_value=key)

        # Contains duplicates -> UID matching problem
        if len(matches) > 1:
            matches_str = options_to_str([store.name for store in matches])
            logger.error(f"Duplicate UID {key} found in stores: {matches_str}")
            raise ValueError(
                f"Duplicate UID `{key}` found in multiple stores: {matches_str}"
            )

        # Get store and remove model
        model_store = matches[0]
        try:
            model_store.remove_model(key)
            logger.info(
                f"Successfully removed model {key} from {model_store.name} store"
            )
        except Exception as error:
            logger.error(
                f"Failed to remove model {key} from store {model_store.name}: {error}"
            )
            raise RuntimeError(
                f"Error occurred while removing model {key} from store `{model_store.name}`"
            ) from error


def clear_model_file_store(*model_types: type[RecordableModel]):
    """Clear all entries from the specified model file stores.

    Args:
        *model_types: One or more TraceableModel types whose stores should be cleared.

    Raises:
        ValueError: If any model_type is not a subclass of TraceableModel.
        KeyError: If any model_type is not found in the store whitelist.
        RuntimeError: If an error occurs while clearing stores.
    """
    logger = _get_logger()
    model_names = [model_type.__name__ for model_type in model_types]
    logger.debug(f"Attempting to clear stores for model types: {model_names}")

    # Input type validation for all model types
    for model_type in model_types:
        if not isinstance(model_type, type) or not issubclass(
            model_type, RecordableModel
        ):
            logger.error(f"Invalid model type: {model_type}")
            raise _expected_got_var_value_error(
                var_name="model_type", got=model_type, expected=type[RecordableModel]
            )

    # Process each model type
    for model_type in model_types:
        # Extract name
        model_name = model_type.__name__
        logger.debug(f"Processing clear for model type: {model_name}")

        # Entry not found in store map
        if model_name not in _STORE_WHITE_LIST_MAP or model_name not in _STORE_MAP:
            logger.error(f"Model type {model_name} not found in store whitelist or map")
            raise _no_entry_key_error(key_value=model_name)

        # Get the store and clear it
        try:
            model_store = _STORE_MAP[model_name]
            items_count = len(model_store)
            model_store.clear()
            logger.info(
                f"Successfully cleared {items_count} items from {model_name} store"
            )
        except Exception as error:
            logger.error(f"Failed to clear store for {model_name}: {error}")
            raise RuntimeError(
                f"Error occurred while clearing store for `{model_name}` type"
            ) from error


# Run when loading module
setup_paths()

if __name__ == "__main__":
    _STORES_DIR = Path("./test_store/")
    if _STORES_DIR.exists():
        shutil.rmtree(_STORES_DIR)
        _STORES_DIR.mkdir()
    else:
        _STORES_DIR.mkdir()
    setup_store()

    # Test path setup functionality
    print("=== Path Setup Test ===")
    print(f"HOME_DIR: {_HOME_DIR}")
    print(f"ASTRO_DIR: {_ASTRO_DIR}")
    print(f"LOG_DIR: {_LOG_DIR}")
    print(f"STORES_DIR: {_STORES_DIR}")
    print(f"Path setup done: {_PATH_SETUP_DONE}")

    # Test logger functionality
    print("\n=== Logger Test ===")
    logger = _get_logger()
    logger.info("Testing logger functionality from paths.py")
    print(f"Logger created: {logger.name}")

    # Test store functionality (if LLMConfig is available)
    print("\n=== Store Test ===")

    from astro.llms import LLMConfig

    # Create a test LLMConfig instance
    test_config = LLMConfig.for_conversational(identifier="ollama")
    print(f"Created test LLMConfig: {test_config.uid}")

    # Save to store
    save_model_file_to_store(test_config)
    print("Saved LLMConfig to store")

    # Load from store
    loaded_config = load_model_file_from_store(test_config.uid)
    print(f"Loaded LLMConfig from store: {loaded_config.uid}")
    print(f"Model names match: {test_config.model_name == type_name(loaded_config)}")

    # Test store clearing
    print(f"Store length before clear: {len(get_model_file_store(LLMConfig))}")
    # clear_model_file_store(LLMConfig)
    print(f"Store length after clear: {len(get_model_file_store(LLMConfig))}")

    print("\n=== Path and Store Test Complete ===")

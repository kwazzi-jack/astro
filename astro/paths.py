import json
import shutil
from collections.abc import ItemsView, Iterator, KeysView, ValuesView
from pathlib import Path
from typing import Any, Generic, TypeAlias

# Lazy logger import to avoid circular dependency
from astro.errors import (
    _expected_got_var_value_error,
    _expected_key_str_value_error,
    _expected_key_type_value_error,
    _expected_value_type_value_error,
    _no_entry_key_error,
)
from astro.typings import (
    ModelType,
    PathDict,
    TraceableModel,
    _options_to_str,
    _path_dict_to_str_dict,
    _str_dict_to_path_dict,
    _type_name,
)

_logger = None


def _get_logger():
    """Get logger instance with lazy initialization to avoid circular imports."""
    global _logger
    if _logger is None:
        from astro.loggings import get_logger

        _logger = get_logger(__file__)
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
    # Use astro/paths.py as reference
    if file_path is None:
        return Path(__file__).parent.parent.resolve()

    # Use specified astro file
    else:
        return Path(file_path).parent.resolve()


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

    # If string file path, convert to Path
    if isinstance(markdown_file_path, str):
        markdown_file_path = Path(markdown_file_path)

    # Validate file exists
    if not markdown_file_path.exists():
        raise FileNotFoundError(f"File `{markdown_file_path}` does not exist")

    # Return contents of markdown file
    try:
        with open(markdown_file_path, encoding="utf-8") as file:
            return file.read().strip()
    # Error while opening file
    except Exception as error:
        raise OSError(
            f"Error while trying to open markdown file '{markdown_file_path}'"
        ) from error


class ModelFileStore(Generic[ModelType]):
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

    def __init__(self, root_dir: Path, model_type: type[ModelType]):
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

        # Create assigned directory if it does not exist
        root_dir.mkdir(exist_ok=True)

        # Validate `model_type` is subclass of `TraceableModel`
        if not (
            isinstance(model_type, type) and issubclass(model_type, TraceableModel)
        ):
            raise _expected_got_var_value_error(
                var_name="model_type",
                got=model_type,
                expected=f"subclass of {TraceableModel}",
            )

        # Attributes
        self._name = root_dir.name
        self._root_dir = root_dir
        self._model_type = model_type
        self._index_file = root_dir / "index"
        self._index_map = {}  # type: PathDict

        # Index file present -> load from file
        if self._index_file.exists():
            self._index_map = self._load_index()

        # No index file -> create clean file
        else:
            self._index_map = {}
            self._save_index()

    @property
    def name(self) -> str:
        """Name of the root directory used for indexing."""
        return self._name

    @property
    def root_dir(self) -> Path:
        """Root directory where model files and index are stored."""
        return self._root_dir

    @property
    def model_type(self) -> type[ModelType]:
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

    def _save_object(self, obj: ModelType):
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
            raise _expected_got_var_value_error(
                var_name="obj", got=type(obj), expected=self.model_type
            )

        try:
            # Fetch file path based on UID
            file_path = self[obj.uid]

            # Dump as json and save (requires as `TraceableModel`)
            with open(file_path, "w") as file:
                json.dump(obj.model_dump(mode="json"), file)

        except Exception as error:
            # An error occurred - propagate up
            raise IOError(
                f"Error while saving `{_type_name(obj)}` ({obj.uid})"
            ) from error

    def _load_object(self, key: str) -> ModelType:
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
            raise _expected_got_var_value_error("key", type(key), str)

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
            raise IOError(f"Error while loading {key}") from error

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
            raise _expected_key_str_value_error(got=type(key))

        # No entry associated with key
        if key not in self:
            raise _no_entry_key_error(key)

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
            raise _expected_key_type_value_error(got=type(key), expected=str)

        if not isinstance(value, Path):
            raise _expected_value_type_value_error(got=type(value), expected=Path)

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
            raise _expected_key_str_value_error(got=type(key))

        # No entry associated with key
        if key not in self:
            raise _no_entry_key_error(key_value=key)

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

    def __contains__(self, key_or_obj: str | ModelType) -> bool:
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
            raise _expected_got_var_value_error(
                var_name="key_or_obj",
                got=type(key_or_obj),
                expected=(str, self.model_type),
            )

        # Return if key or UID has associated entry
        if isinstance(key_or_obj, str):
            return key_or_obj in self._index_map
        else:
            return key_or_obj.uid in self._index_map

    def get_model(self, key: str, default: Any = None) -> ModelType | Any:
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
            except IOError:
                return default

        # Key not found, return default
        return default

    def add_model(self, obj: ModelType):
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
            raise _expected_value_type_value_error(
                got=type(obj), expected=self.model_type
            )

        # Create file path based on input object
        file_path = self.root_dir / obj.uid

        # Set object based on UID and file path
        self.__setitem__(obj.uid, file_path)

        # Update index file
        self._save_object(obj)
        self._save_index()

    def remove_model(self, key_or_obj: str | ModelType):
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
            raise _expected_got_var_value_error(
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
ModelTypeDict: TypeAlias = dict[str, type[TraceableModel]]
StoreDict: TypeAlias = dict[str, ModelFileStore[ModelType]]

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

    global _STORE_WHITE_LIST_MAP

    # List of TraceableModels to white list
    # IMPORTANT - B - This governs which models can be stored
    from astro.llms import LLMConfig

    # Non-empty list will not get re-created
    # IMPORTANT - B - restart Astro instead
    if len(_STORE_WHITE_LIST_MAP):
        return

    # Create list based on above
    _STORE_WHITE_LIST_MAP = {LLMConfig.__name__: LLMConfig}


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

    # Inpute type validation
    if not isinstance(dir_path, Path):
        raise _expected_got_var_value_error(
            var_name="dir_path", got=type(dir_path), expected=Path
        )

    # If not in white-list -> False
    if dir_path.name not in _STORE_WHITE_LIST_MAP:
        return False

    # If directory doesn't exist or not a directory -> False
    if not dir_path.exists() or not dir_path.is_dir():
        return False

    # Directory contents
    dir_contents = list(dir_path.iterdir())

    # If empty directory -> False
    if len(dir_contents) == 0:
        return False

    # If no index file -> False
    if not any(item.name == "index" for item in dir_contents):
        return False

    # Assume true
    return True


def _setup_store_map():
    global _STORE_MAP

    # Issue with missing path
    if not _PATH_SETUP_DONE or _STORES_DIR is None:
        raise RuntimeError("Either pathing is not setup or stores directory is not set")

    # Only setup once
    if len(_STORE_MAP) != 0:
        return

    # Get list of stores
    for store_path in _STORES_DIR.iterdir():
        # Is it a store directory -> setup
        if _is_store_dir(store_path):
            # Get model type, name and then store
            model_name = store_path.name
            model_type = _STORE_WHITE_LIST_MAP[model_name]
            model_store = ModelFileStore[model_type](
                root_dir=store_path, model_type=model_type
            )

            # Add model store to map
            _STORE_MAP[model_store.name] = model_store

        # If not a store, remove
        # Is directory -> delete contents and remove
        elif store_path.is_dir():
            shutil.rmtree(store_path)
            store_path.rmdir()

        # Is file -> delete
        else:
            store_path.unlink()

    # Ensure all white-listed model types have a store
    for model_name, model_type in _STORE_WHITE_LIST_MAP.items():
        if model_name not in _STORE_MAP:
            store_path = _STORES_DIR / model_name
            model_store = ModelFileStore[model_type](
                root_dir=store_path, model_type=model_type
            )
            _STORE_MAP[model_store.name] = model_store


def setup_store():
    """Set up the model file store system.

    This is separate from path setup and handles TraceableModel serialization.

    Raises:
        RuntimeError: If paths have not been initialized via setup_paths().
    """
    # Check if paths are initialized
    if _ASTRO_DIR is None:
        raise RuntimeError(
            "Paths not initialized. Call setup_paths() before setup_store()."
        )

    # Load white list
    _load_store_white_list()

    # Load store map
    # CAUTION - B - Deletes non-white listed model stores
    _setup_store_map()


def get_model_file_store(model_type: type[TraceableModel]) -> ModelFileStore:
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
    # Input type validation
    if not isinstance(model_type, type) or not issubclass(model_type, TraceableModel):
        raise _expected_got_var_value_error(
            var_name="model_type", got=model_type, expected=type[TraceableModel]
        )
    # Extract name
    model_name = model_type.__name__

    # Entry not found in store map
    # B - Might be redundant double check
    if model_name not in _STORE_WHITE_LIST_MAP or model_name not in _STORE_MAP:
        raise _no_entry_key_error(key_value=model_name)

    # Return correspondoning model file store
    return _STORE_MAP[model_name]


def save_model_file_to_store(model_file: TraceableModel):
    # Extract input type
    model_type = type(model_file)

    # Input type validation
    if not isinstance(model_file, TraceableModel):
        raise _expected_got_var_value_error(
            var_name="model_type", got=model_type, expected=TraceableModel
        )

    # Not a valid model file to store
    if model_type.__name__ not in _STORE_WHITE_LIST_MAP:
        raise _no_entry_key_error(key_value=model_type.__name__)

    try:
        # Get model file store
        model_store = get_model_file_store(model_type)

        # Add new model to store
        model_store.add_model(model_file)

    # Error occurred while trying to load file store and save model
    except Exception as error:
        raise RuntimeError(
            f"Error occurred while saving model {model_file.uid} to store for `{model_type.__name__}` type"
        ) from error


def load_model_file_from_store(key: str) -> TraceableModel:
    # Input type validation
    if not isinstance(key, str):
        raise _expected_key_str_value_error(got=type(key))

    # Check stores for entry
    matches: list[ModelFileStore] = []
    for store in _STORE_MAP.values():
        # If entry in store -> add match
        if key in store:
            matches.append(store)

    # No matches
    if len(matches) == 0:
        raise _no_entry_key_error(key_value=key)

    # Contains duplicates -> UID matching problem
    if len(matches) > 1:
        matches_str = _options_to_str([store.name for store in matches])
        raise ValueError(
            f"Duplicate UID `{key}` found in multiple stores: {matches_str}"
        )

    # Get store
    model_store = matches[0]
    try:
        # Return model from model store
        return model_store.get_model(key)
    except Exception as error:
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
    # Input type validation for all keys
    for key in keys:
        if not isinstance(key, str):
            raise _expected_key_str_value_error(got=type(key))

    # Process each key
    for key in keys:
        # Check stores for entry
        matches: list[ModelFileStore] = []
        for store in _STORE_MAP.values():
            # If entry in store -> add match
            if key in store:
                matches.append(store)

        # No matches
        if len(matches) == 0:
            raise _no_entry_key_error(key_value=key)

        # Contains duplicates -> UID matching problem
        if len(matches) > 1:
            matches_str = _options_to_str([store.name for store in matches])
            raise ValueError(
                f"Duplicate UID `{key}` found in multiple stores: {matches_str}"
            )

        # Get store and remove model
        model_store = matches[0]
        try:
            model_store.remove_model(key)
        except Exception as error:
            raise RuntimeError(
                f"Error occurred while removing model {key} from store `{model_store.name}`"
            ) from error


def clear_model_file_store(*model_types: type[TraceableModel]):
    """Clear all entries from the specified model file stores.

    Args:
        *model_types: One or more TraceableModel types whose stores should be cleared.

    Raises:
        ValueError: If any model_type is not a subclass of TraceableModel.
        KeyError: If any model_type is not found in the store whitelist.
        RuntimeError: If an error occurs while clearing stores.
    """
    # Input type validation for all model types
    for model_type in model_types:
        if not isinstance(model_type, type) or not issubclass(
            model_type, TraceableModel
        ):
            raise _expected_got_var_value_error(
                var_name="model_type", got=model_type, expected=type[TraceableModel]
            )

    # Process each model type
    for model_type in model_types:
        # Extract name
        model_name = model_type.__name__

        # Entry not found in store map
        if model_name not in _STORE_WHITE_LIST_MAP or model_name not in _STORE_MAP:
            raise _no_entry_key_error(key_value=model_name)

        # Get the store and clear it
        try:
            model_store = _STORE_MAP[model_name]
            model_store.clear()
        except Exception as error:
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
    print(f"Model names match: {test_config.model_name == _type_name(loaded_config)}")

    # Test store clearing
    print(f"Store length before clear: {len(get_model_file_store(LLMConfig))}")
    # clear_model_file_store(LLMConfig)
    print(f"Store length after clear: {len(get_model_file_store(LLMConfig))}")

    print("\n=== Path and Store Test Complete ===")

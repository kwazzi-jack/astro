from pathlib import Path

# General Path Type
StrPath = str | Path

# Astro's home
HOME_DIR = Path.home()
ASTRO_DIR = HOME_DIR / ".astro"
ASTRO_DIR.mkdir(exist_ok=True)

# Path for common conversations database
CONV_DB_PATH = ASTRO_DIR / "conversations.sqlite"

# Path for logs
LOG_DIR = ASTRO_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Path for configs
BASE_ENV_PATH = ASTRO_DIR / ".secrets"


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


# Installation home
INSTALL_HOME_DIR = get_module_dir()


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
        raise FileNotFoundError(f"File '{markdown_file_path}' does not exist")

    # Return contents of markdown file
    try:
        with open(markdown_file_path, encoding="utf-8") as file:
            return file.read().strip()
    # Error while opening file
    except Exception as error:
        raise OSError(
            f"Error while trying to open markdown file '{markdown_file_path}'"
        ) from error


if __name__ == "__main__":
    print(f"{INSTALL_HOME_DIR=}")

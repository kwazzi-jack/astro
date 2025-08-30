import zlib
from pathlib import Path

from dotenv import dotenv_values
from pydantic import SecretStr

from astro.loggings.base import get_logger
from astro.paths import _ASTRO_DIR, _BASE_SECRETS_PATH

logger = get_logger("astro.utilities.security")


def checksum(
    path: str | Path,
    chunk_size: int = -1,
) -> int:
    if isinstance(path, str):
        # Convert str to Path
        path = Path(path)

    if not path.exists():
        # File does not exist
        msg = f"File does not exist: `{path}`"
        logger.error(msg)
        raise FileNotFoundError(msg)

    # Run checksum algorithm
    total = 0
    logger.debug(f"Running checksum on '{path.name}' with `{chunk_size=}`")
    with open(path, "rb") as file:
        # Run with chunks if > 0
        while chunk := file.read(chunk_size):
            total = zlib.crc32(chunk, total)

    # Return checksum
    logger.debug(f"Checksum complete: {total}")
    return total


def files_differ(file1: str | Path, file2: str | Path, chunk_size: int = -1) -> bool:
    files_equal = checksum(file1, chunk_size) == checksum(file2, chunk_size)
    # FIXME logger.debug(f"File '{file1.name}' == File '{file2.name}'? {files_equal}")
    return not files_equal


def get_secret_key(key: str) -> SecretStr | None:
    if not _BASE_SECRETS_PATH.exists():
        error_msg = (
            "Cannot find environment file at `~/.astro`. "
            "#TODO - B - Add functionality to add keys dynamically"
        )
        raise ValueError(error_msg)

    env_dict = dotenv_values(_BASE_SECRETS_PATH)
    if len(env_dict) == 0 or key not in env_dict:
        raise ValueError(f"Environment file empty or missing key `{key}`")

    value = env_dict[key]
    if not isinstance(value, str):
        raise ValueError(
            f"Environment key `{value}` has a non-string value (`{type(value).__class__}`)"
        )
    return SecretStr(value)


if __name__ == "__main__":
    from astro.utilities.display import rprint

    rprint(f"`{get_secret_key("KEY").get_secret_value()=}`")

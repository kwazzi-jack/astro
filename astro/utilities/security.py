import json
import zlib
from pathlib import Path
from typing import Any

import blake3
from dotenv import dotenv_values
from pydantic import SecretStr

from astro.loggings.base import get_loggy
from astro.paths import SECRETS_PATH

loggy = get_loggy("astro.utilities.security")


def checksum(
    path: str | Path,
    chunk_size: int = -1,
) -> int:
    if isinstance(path, str):
        # Convert str to Path
        path = Path(path)

    if not path.exists():
        # File does not exist
        raise loggy.FileNotFoundError(f"File does not exist: `{path}`")

    # Run checksum algorithm
    total = 0
    loggy.debug(f"Running checksum on '{path.name}' with `{chunk_size=}`")
    with open(path, "rb") as file:
        # Run with chunks if > 0
        while chunk := file.read(chunk_size):
            total = zlib.crc32(chunk, total)

    # Return checksum
    loggy.debug(f"Checksum complete: {total}")
    return total


def files_differ(file1: str | Path, file2: str | Path, chunk_size: int = -1) -> bool:
    files_equal = checksum(file1, chunk_size) == checksum(file2, chunk_size)
    return not files_equal


def get_secret_key(key: str) -> SecretStr:
    # Input validation
    if SECRETS_PATH is None:
        raise loggy.FileNotFoundError(
            "Secrets file has not been set. Ensure one is present at `$ASTRO_HOME`. "
            "#TODO - Add functionality to add keys dynamically"
        )

    if not isinstance(key, str):
        raise

    if not SECRETS_PATH.exists():
        raise loggy.FileNotFoundError(
            "Secrets file cannot be found. Ensure one is present at `$ASTRO_HOME`. "
            "#TODO - Add functionality to add keys dynamically"
        )

    secrets_dict = dotenv_values(SECRETS_PATH)
    if len(secrets_dict) == 0:
        raise ValueError("Secrets file is empty")

    if key not in secrets_dict:
        raise loggy.NoEntryError(key_value=key)

    secret_value = secrets_dict[key]
    if not isinstance(secret_value, str):
        raise loggy.ExpectedVariableType(
            var_name=key, got=type(secret_value), expected=str
        )

    return SecretStr(secret_value)


def _compute_stable_hash_from_dict(obj_dict: dict[Any, Any]) -> bytes:
    # Create deterministic JSON string
    # Brian If something is wrong with hashing, its probably here
    json_str = json.dumps(
        obj_dict,
        sort_keys=True,  # IMPORTANT: for consistent key order
        separators=(",", ";"),  # No whitespace
        ensure_ascii=True,  # Avoid encoding variations
        default=str,  # Fallback conversion
    )

    # Generate blake3 hash
    return blake3.blake3(json_str.encode("utf-8")).digest()


if __name__ == "__main__":
    from astro.utilities.display import rprint

    rprint(f"`{get_secret_key("KEY").get_secret_value()=}`")

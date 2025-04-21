from pathlib import Path
from typing import Iterator
import appdirs

# Create cache directory for astro
CACHE_DIR = Path(appdirs.user_cache_dir(appname="astro"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Path for common messages database
MESSAGE_DB_PATH = CACHE_DIR / "messages.sqlite"

# Path for common data repository
DATA_DIR = CACHE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
IN_DIR = DATA_DIR / "inputs"
IN_DIR.mkdir(exist_ok=True)
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# Path for conversations
CONV_DIR = CACHE_DIR / "conversations"
CONV_DIR.mkdir(exist_ok=True)
GLOBAL_DIR = CONV_DIR / "global"
GLOBAL_DIR.mkdir(exist_ok=True)

# Path for logs
LOG_DIR = CACHE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def is_conv_dir(item: Path) -> bool:
    if item.is_dir() and (item.name.startswith("conv") or item.name == "global"):
        return True
    return False


def list_cache() -> Iterator[Path]:
    return CACHE_DIR.glob("*")


def get_conv_dir(conv_id: str) -> Path | None:
    if conv_id == "global":
        return GLOBAL_DIR

    for item in CONV_DIR.glob("*"):
        if is_conv_dir(item) and item.name == conv_id:
            return item
    return None

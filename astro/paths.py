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
BASE_ENV_PATH = ASTRO_DIR / ".env"


def get_module_dir(file: str | None = None) -> Path:
    if file is None:
        return Path(__file__).parent.parent.resolve()
    else:
        return Path(file).parent.resolve()

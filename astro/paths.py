from pathlib import Path

# Astro's home
ASTRO_DIR = Path.home() / ".astro"
ASTRO_DIR.mkdir(exist_ok=True)

# Path for common conversations database
CONV_DB_PATH = ASTRO_DIR / "conversations.sqlite"

# Path for logs
LOG_DIR = ASTRO_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_module_dir(file: str | None = None) -> Path:
    if file is None:
        return Path(__file__).parent.parent
    else:
        return Path(file).parent

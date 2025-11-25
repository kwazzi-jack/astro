# --- Internal Imports ---
from importlib import metadata

# --- Local Imports ---
from astro.__version__ import version
from astro.typings.base import NamedDict


def get_astro_version() -> str:
    """Retrieve the version of the astro package.

    Returns:
        str: The version string of the astro package.
    """
    return version


def get_local_package_version(package: str = "astro") -> str:
    """Get the version of a local package, defaulting to astro.

    Args:
        package (str): Name of the package to query. Defaults to "astro".

    Returns:
        str: The version string of the package, or "NaN" if not found.
    """
    if package == "astro":
        return get_astro_version()
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "NaN"


def get_build_metadata() -> NamedDict:
    """Retrieve build metadata for the astro package.

    Returns:
        NamedDict: A dictionary containing version and build date.
    """
    data = metadata.metadata("astro")
    return {
        "version": data["Version"],
        "build_date": data.get("Build-Date", "unknown"),
    }

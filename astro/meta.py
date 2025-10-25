from importlib import metadata

from astro.__version__ import version


def get_astro_version() -> str:
    return version


def get_local_package_version(package: str = "astro") -> str:
    if package == "astro":
        return get_astro_version()
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "NaN"


def get_build_meta():
    data = metadata.metadata("astro")
    return {
        "version": data["Version"],
        "build_date": data.get("Build-Date", "unknown"),
    }

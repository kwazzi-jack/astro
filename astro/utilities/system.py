import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import distro
from pydantic import BaseModel, Field


def get_users_name() -> str:
    return os.getlogin()


def is_exe_installed(exe_name: str) -> bool:
    return shutil.which(exe_name) is not None


def _check_dockerenv_file() -> bool:
    """Check for Docker container indicator file."""
    return Path("/.dockerenv").exists()


def _check_proc_cgroup_docker() -> bool:
    """Check /proc/1/cgroup for Docker container indicators."""
    cgroup_path = Path("/proc/1/cgroup")
    if not cgroup_path.exists():
        return False

    try:
        content = cgroup_path.read_text()
        return "docker" in content
    except (OSError, IOError):
        return False


def _check_container_env() -> bool:
    """Check for generic container environment variable."""
    return os.environ.get("container") is not None


def _check_kubernetes_secrets() -> bool:
    """Check for Kubernetes service account secrets."""
    return Path("/run/secrets/kubernetes.io").exists()


def is_containerized() -> bool:
    """
    Detect containerized environment using multiple detection methods.

    Returns:
        True if running in a detected container environment
    """
    container_checks = [
        _check_dockerenv_file,
        _check_proc_cgroup_docker,
        _check_container_env,
        _check_kubernetes_secrets,
    ]

    for check in container_checks:
        try:
            if check():
                return True
        except Exception:
            pass

    return False


def _check_uname_for_microsoft() -> bool:
    """Check if uname release contains Microsoft signature."""
    return "microsoft" in platform.uname().release.lower()


def _check_uname_for_wsl() -> bool:
    """Check if uname release contains WSL signature."""
    return "wsl" in platform.uname().release.lower()


def _check_proc_version_for_wsl() -> bool:
    """Check /proc/version for Microsoft WSL indicators."""
    proc_version_path = Path("/proc/version")
    if not proc_version_path.exists():
        return False

    try:
        content = proc_version_path.read_text().lower()
        return any(indicator in content for indicator in ["microsoft", "wsl"])
    except (OSError, IOError):
        return False


def _check_wsl_distro_env() -> bool:
    """Check for WSL_DISTRO_NAME environment variable."""
    return os.environ.get("WSL_DISTRO_NAME") is not None


def _check_wslenv() -> bool:
    """Check for WSLENV environment variable."""
    return os.environ.get("WSLENV") is not None


def is_wsl() -> bool:
    """
    Detect Windows Subsystem for Linux environment using multiple indicators.

    Returns:
        True if running in WSL, False otherwise
    """
    wsl_checks = [
        _check_uname_for_microsoft,
        _check_uname_for_wsl,
        _check_proc_version_for_wsl,
        _check_wsl_distro_env,
        _check_wslenv,
    ]

    for check in wsl_checks:
        try:
            if check():
                return True
        except Exception:
            pass

    return False


class PlatformDetails(BaseModel):
    """Pydantic model for system information with validation."""

    platform: str
    is_wsl: bool = Field(default_factory=is_wsl)
    is_containerized: bool = Field(default_factory=is_containerized)
    distribution: str | None
    kernel: str | None
    architecture: str


def _get_windows_details(architecture: str) -> PlatformDetails:
    """
    Extract Windows-specific system information.

    Args:
        architecture: System hardware architecture

    Returns:
        SystemDetails model with Windows platform information
    """
    return PlatformDetails(
        platform="windows",
        is_wsl=False,
        is_containerized=is_containerized(),
        distribution=f"Windows {platform.release()} {platform.version()}",
        kernel=platform.version(),
        architecture=architecture,
    )


def _get_macos_details(architecture: str) -> PlatformDetails:
    """
    Extract macOS-specific system information.

    Args:
        architecture: System hardware architecture

    Returns:
        SystemDetails model with macOS platform information
    """
    try:
        mac_version, _, _ = platform.mac_ver()
        distribution = f"macOS {mac_version}" if mac_version else "macOS"
    except Exception:
        distribution = "macOS"

    return PlatformDetails(
        platform="macos",
        is_wsl=False,
        distribution=distribution,
        kernel=platform.release(),
        architecture=architecture,
    )


def _read_os_release() -> str | None:
    """
    Parse /etc/os-release for distribution information.

    Returns:
        Distribution string from PRETTY_NAME field or None
    """
    os_release_paths = [Path("/etc/os-release"), Path("/usr/lib/os-release")]

    for path in os_release_paths:
        if not path.exists():
            continue

        try:
            content = path.read_text()
            for line in content.splitlines():
                if line.startswith("PRETTY_NAME="):
                    return line.split("=", 1)[1].strip("\"'")
        except (OSError, IOError):
            continue

    return None


def _read_lsb_release() -> str | None:
    """
    Execute lsb_release command for distribution detection.

    Returns:
        Distribution description string or None
    """
    try:
        result = subprocess.run(
            ["lsb_release", "-d"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )

        if result.returncode == 0 and result.stdout:
            description = result.stdout.strip()
            if ":" in description:
                return description.split(":", 1)[1].strip()
            return description

    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return None


def _check_distribution_files() -> str | None:
    """
    Check distribution-specific release files for identification.

    Returns:
        Distribution name with version or None
    """
    distro_files = {
        "/etc/redhat-release": "Red Hat",
        "/etc/debian_version": "Debian",
        "/etc/centos-release": "CentOS",
        "/etc/fedora-release": "Fedora",
        "/etc/arch-release": "Arch Linux",
        "/etc/alpine-release": "Alpine Linux",
        "/etc/gentoo-release": "Gentoo",
        "/etc/slackware-version": "Slackware",
    }

    for file_path, distro_name in distro_files.items():
        path = Path(file_path)
        if not path.exists():
            continue

        try:
            content = path.read_text().strip()
            if content and content != distro_name:
                return f"{distro_name} {content}"
            return distro_name
        except (OSError, IOError):
            return distro_name

    return None


def _platform_fallback() -> str:
    """
    Fallback to generic Linux identification using platform module.

    Returns:
        Generic Linux identification string
    """
    return f"Linux {platform.release()}"


def _get_distro_package_info() -> str | None:
    """
    Use distro package for accurate Linux distribution detection.

    Returns:
        Formatted distribution string or None if detection fails
    """
    try:
        name = distro.name()
        version = distro.version()
        codename = distro.codename()

        if not name:
            return None

        parts = [name]
        if version:
            parts.append(version)
        if codename:
            parts.append(f"({codename})")

        return " ".join(parts)
    except Exception:
        return None


def _get_linux_distribution() -> str:
    """
    Determine Linux distribution using distro package with fallback methods.

    Returns:
        Formatted distribution string with version information
    """
    # Primary method: distro package
    distro_info = _get_distro_package_info()
    if distro_info:
        return distro_info

    # Fallback detection methods
    fallback_methods = [
        _read_os_release,
        _read_lsb_release,
        _check_distribution_files,
        _platform_fallback,
    ]

    for method in fallback_methods:
        try:
            result = method()
            if result:
                return result
        except Exception:
            pass

    return "Linux (unknown distribution)"


def _get_linux_details(architecture: str) -> PlatformDetails:
    """
    Extract Linux distribution information with WSL detection.

    Args:
        architecture: System hardware architecture

    Returns:
        SystemDetails model with Linux platform information
    """
    distribution = _get_linux_distribution()

    return PlatformDetails(
        platform="linux",
        distribution=distribution,
        kernel=platform.release(),
        architecture=architecture,
    )


def get_platform_details() -> PlatformDetails:
    """
    Detect comprehensive operating system and platform information.

    Returns:
        SystemDetails model containing validated system information:
        - platform: Primary platform identifier
        - is_wsl: WSL environment detection flag
        - distribution: OS distribution name or version
        - kernel: Kernel version information
        - architecture: System hardware architecture
    """
    base_system = platform.system().lower()
    architecture = platform.machine()

    match base_system:
        case "windows":
            return _get_windows_details(architecture)
        case "darwin":
            return _get_macos_details(architecture)
        case "linux":
            return _get_linux_details(architecture)
        case _:
            return PlatformDetails(
                platform="unknown",
                distribution=None,
                kernel=platform.release(),
                architecture=architecture,
            )


def get_platform_str(platform_info: PlatformDetails) -> str:
    # Add platform
    result: str = platform_info.platform.capitalize()

    # Add WSL or Containerized if applicable
    if platform_info.is_wsl and platform_info.is_containerized:
        result += " (WSL + Containerized)"
    elif platform_info.is_wsl:
        result += " (WSL)"
    elif platform_info.is_containerized:
        result += " (Containerized)"

    # Add distro if applicable
    if platform_info.distribution is not None:
        result += f"; Distribution: {platform_info.distribution.capitalize()}"

    # Add kernel if applicable
    if platform_info.kernel is not None:
        result += f"; Kernel: `{platform_info.kernel}`"

    # Add architecture
    result += f"; Architecture: `{platform_info.architecture}`"

    # Return platform string
    return result


# Python Environment Detection
def _check_uv_environment() -> tuple[bool, str | None]:
    """Check for UV (Astral) package manager environment."""
    # Check for UV environment variable
    if os.environ.get("UV_PROJECT_ENVIRONMENT"):
        return True, "uv (via UV_PROJECT_ENVIRONMENT)"

    # Check for uv.lock in current directory
    if (Path.cwd() / "uv.lock").exists():
        return True, "uv (uv.lock detected)"

    # Check if running in UV-managed venv
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path and Path(venv_path).name == ".venv":
        # Check if parent has uv.lock
        try:
            parent = Path(venv_path).parent
            if (parent / "uv.lock").exists():
                return True, "uv (.venv with uv.lock)"
        except Exception:
            pass

    # Check for executable
    if is_exe_installed("uv") or is_exe_installed("uvx"):
        return True, "uv (executable found)"

    return False, None


def _check_poetry_environment() -> tuple[bool, str | None]:
    """Check for Poetry environment."""
    # Check Poetry environment variable
    if os.environ.get("POETRY_ACTIVE"):
        return True, "poetry (POETRY_ACTIVE)"

    # Check for poetry.lock in current directory
    if (Path.cwd() / "poetry.lock").exists():
        return True, "poetry (poetry.lock detected)"

    # Check for executable
    if is_exe_installed("poetry"):
        return True, "poetry (executable found)"

    return False, None


def _check_conda_environment() -> tuple[bool, str | None]:
    """Check for Conda environment."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        return True, f"conda ({conda_env})"

    # Check for conda-meta directory
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix and Path(conda_prefix, "conda-meta").exists():
        return True, "conda (conda-meta detected)"

    # Check for executable
    if is_exe_installed("conda"):
        return True, "conda (executable found)"

    return False, None


def _check_pyenv_environment() -> tuple[bool, str | None]:
    """Check for Pyenv environment."""
    # Check pyenv environment variable
    pyenv_version = os.environ.get("PYENV_VERSION")
    if pyenv_version:
        return True, f"pyenv ({pyenv_version})"

    # Check for executable
    if is_exe_installed("pyenv"):
        return True, "pyenv (executable found)"

    return False, None


def _check_venv_environment() -> tuple[bool, str | None]:
    """Check for standard venv environment."""
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path is None:
        return False, None

    venv_name = Path(venv_path).name

    # Check for pyvenv.cfg
    pyvenv_cfg = Path(venv_path, "pyvenv.cfg")
    if pyvenv_cfg.exists():
        return True, f"venv ({venv_name})"

    return True, f"venv-like ({venv_name})"


def _check_pipenv_environment() -> tuple[bool, str | None]:
    """Check for Pipenv environment."""
    if os.environ.get("PIPENV_ACTIVE") is not None:
        return True, "pipenv (PIPENV_ACTIVE)"

    # Check for Pipfile in current directory
    if (Path.cwd() / "Pipfile").exists():
        return True, "pipenv (Pipfile detected)"

    # Check for executable
    if is_exe_installed("pipenv"):
        return True, "pipenv (executable found)"

    return False, None


def _get_python_version() -> str:
    """Get current Python version."""
    version_info = sys.version_info
    return f"{version_info.major}.{version_info.minor}.{version_info.micro}"


class PythonEnvironmentDetails(BaseModel):
    """Pydantic model for Python environment information."""

    version: str
    environment_type: str
    environment_details: str | None = None
    virtual_env_active: bool = False
    virtual_env_name: str | None = None
    potential_issues: list[str] = Field(default_factory=list)


def get_python_details() -> PythonEnvironmentDetails:
    """
    Detect comprehensive Python environment information.

    Returns:
        PythonEnvironmentDetails model containing:
        - version: Python version string
        - environment_type: Type of Python environment
        - environment_details: Additional details about the environment
        - virtual_env_active: Whether a virtual environment is active
        - virtual_env_name: Name of the virtual environment if active
        - potential_issues: List of detected issues
    """
    version = _get_python_version()

    # Check environment types in order of specificity
    environment_checks = [
        ("uv", _check_uv_environment),
        ("poetry", _check_poetry_environment),
        ("conda", _check_conda_environment),
        ("pipenv", _check_pipenv_environment),
        ("pyenv", _check_pyenv_environment),
        ("venv", _check_venv_environment),
    ]

    for env_name, check_func in environment_checks:
        try:
            is_active, details = check_func()
            if is_active:
                venv_path = os.environ.get("VIRTUAL_ENV")
                venv_name = Path(venv_path).name if venv_path else None

                return PythonEnvironmentDetails(
                    version=version,
                    environment_type=env_name,
                    environment_details=details,
                    virtual_env_active=bool(venv_path),
                    virtual_env_name=venv_name,
                )
        except Exception:
            continue

    # Default to native Python
    return PythonEnvironmentDetails(
        version=version,
        environment_type="native",
        environment_details="System Python installation",
        virtual_env_active=False,
        virtual_env_name=None,
    )


def get_python_environment_str(python_info: PythonEnvironmentDetails) -> str:
    """Format Python environment information as a concise string."""
    result = f"`python{python_info.version}`"

    # Add environment type
    if python_info.environment_type != "native":
        result += f" (via `{python_info.environment_type}`"
        if python_info.virtual_env_active and python_info.virtual_env_name:
            result += f": `{python_info.virtual_env_name}`"
        result += ")"

    # Add issues if any (but keep brief for privacy)
    if (
        python_info.potential_issues is not None
        and len(python_info.potential_issues) > 0
    ):
        issue_count = len(python_info.potential_issues)
        result += f" [{issue_count} potential issue{'s' if issue_count > 1 else ''}]"

    return result


def get_python_version_str(python_info: PythonEnvironmentDetails) -> str:
    return python_info.version


# Example usage and testing
if __name__ == "__main__":
    from astro.utilities.display import rprint

    platform_info = get_platform_details()
    python_info = get_python_details()

    rprint(platform_info)
    rprint(python_info)

    print(f"Platform: {get_platform_str(platform_info)}")
    print(f"Python: {get_python_environment_str(python_info)}")

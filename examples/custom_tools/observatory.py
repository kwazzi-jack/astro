"""Domain objects shared by the custom tool examples."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass(slots=True)
class Observatory:
    """Describe an observatory used by helper tools.

    Attributes:
        name (str): Friendly name of the site.
        latitude_deg (float): Latitude in decimal degrees.
        longitude_deg (float): Longitude in decimal degrees.
        instruments (list[str]): Available instrument names at the site.
        site_tags (list[str]): Lightweight descriptors used for chat context.
    """

    name: str
    latitude_deg: float
    longitude_deg: float
    instruments: list[str]
    site_tags: list[str] = field(default_factory=list)

    def slug(self) -> str:
        """Return a slugified identifier derived from the friendly name.

        Returns:
            str: Lowercase slug containing letters, digits, and hyphens.

        Examples:
            >>> Observatory(
            ...     name="Andromeda Station",
            ...     latitude_deg=0,
            ...     longitude_deg=0,
            ...     instruments=[],
            ... ).slug()
            'andromeda-station'
        """

        normalized = self.name.strip().lower()
        slug_value = normalized.replace(" ", "-")
        return slug_value

    def summary(self) -> str:
        """Return a short sentence describing the observatory.

        Returns:
            str: Description including coordinates and notable tags.

        Examples:
            >>> observatory = Observatory(
            ...     name="Andromeda Station",
            ...     latitude_deg=34.2,
            ...     longitude_deg=-118.1,
            ...     instruments=["WideCam"],
            ... )
            >>> observatory.summary()
            'Andromeda Station at lat 34.2 deg, lon -118.1 deg (tags: none)'
        """

        tag_text = ", ".join(self.site_tags) if self.site_tags else "none"
        return (
            f"{self.name} at lat {self.latitude_deg} deg, "
            f"lon {self.longitude_deg} deg (tags: {tag_text})"
        )

    def instrument_table(self) -> str:
        """Return a comma separated list of instruments.

        Returns:
            str: Concise summary of available instruments.

        Examples:
            >>> observatory = Observatory(
            ...     name="Andromeda Station",
            ...     latitude_deg=34.2,
            ...     longitude_deg=-118.1,
            ...     instruments=["WideCam", "SpecOne"],
            ... )
            >>> observatory.instrument_table()
            'WideCam, SpecOne'
        """

        if not self.instruments:
            return "(no registered instruments)"
        return ", ".join(self.instruments)


_DEFAULT_INSTRUMENTS = [
    "WideCam-3 for wide-field imaging",
    "SpecOne high-resolution spectrograph",
    "CloudWatcher all-sky monitor",
]

_DEFAULT_TAGS = [
    "desert site",
    "robotic",
    "low humidity",
]


def create_default_observatory(
    instruments: Sequence[str] | None = None,
    site_tags: Sequence[str] | None = None,
) -> Observatory:
    """Create the stock observatory used by the custom tools.

    Args:
        instruments (Sequence[str] | None): Optional override list of
            instruments to register.
        site_tags (Sequence[str] | None): Optional override of descriptive tags.

    Returns:
        Observatory: Configured observatory descriptor.

    Examples:
        >>> observatory = create_default_observatory()
        >>> observatory.name
        'Astro Demo Station'
    """

    inst_list = (
        list(instruments) if instruments is not None else list(_DEFAULT_INSTRUMENTS)
    )
    tag_list = list(site_tags) if site_tags is not None else list(_DEFAULT_TAGS)
    return Observatory(
        name="Astro Demo Station",
        latitude_deg=34.2,
        longitude_deg=-118.1,
        instruments=inst_list,
        site_tags=tag_list,
    )


def create_reference_observatories() -> dict[str, Observatory]:
    """Create the initial map of named observatories referenced by tools.

    Returns:
        dict[str, Observatory]: Mapping of slug -> observatory descriptor.
    """

    desert_site = create_default_observatory()
    alpine_site = Observatory(
        name="Summit Array",
        latitude_deg=46.2,
        longitude_deg=8.0,
        instruments=[
            "SummitCam wide-field imager",
            "IRSpec infrared spectrograph",
        ],
        site_tags=["alpine", "adaptive optics", "cryogenic"],
    )
    coastal_site = Observatory(
        name="Harbor Scope",
        latitude_deg=-33.9,
        longitude_deg=151.2,
        instruments=[
            "SeaWatcher radar",
            "OptiTrack optical tracker",
            "MetStation weather array",
        ],
        site_tags=["coastal", "tracking", "weather"],
    )
    return {
        desert_site.slug(): desert_site,
        alpine_site.slug(): alpine_site,
        coastal_site.slug(): coastal_site,
    }


REFERENCE_OBSERVATORIES: dict[str, Observatory] = create_reference_observatories()


def register_observatories(sites: Iterable[Observatory]) -> None:
    """Add or replace observatories in the shared registry.

    Args:
        sites (Iterable[Observatory]): Observatories to register.

    Returns:
        None: The registry is updated in place.
    """

    for site in sites:
        slug = site.slug()
        REFERENCE_OBSERVATORIES[slug] = site


def _pick_observatory(slug: str | None) -> Observatory:
    if slug is None:
        # First item is the default demo site
        return next(iter(REFERENCE_OBSERVATORIES.values()))
    try:
        return REFERENCE_OBSERVATORIES[slug]
    except KeyError as error:
        raise ValueError("Unknown observatory requested") from error


def list_sites() -> str:
    """Return a newline-delimited catalog of registered observatories.

    Returns:
        str: Catalog including coordinates and feature tags for each site.
    """

    lines = [
        "Registered observatories:",
        "------------------------",
    ]
    for slug, site in REFERENCE_OBSERVATORIES.items():
        lines.append(f"{slug}: {site.summary()}")
    return "\n".join(lines)


def describe_site(slug: str | None = None) -> str:
    """Return a paragraph describing the requested observatory.

    Args:
        slug (str | None): Observatory slug to describe. Defaults to the demo site.

    Returns:
        str: Readable description containing coordinates and tags.
    """

    site = _pick_observatory(slug)
    return site.summary()


def list_instruments(slug: str | None = None) -> str:
    """Return the available instruments with short annotations.

    Args:
        slug (str | None): Observatory slug to inspect. Defaults to the demo site.

    Returns:
        str: Comma separated list of instrument descriptors.
    """

    site = _pick_observatory(slug)
    return site.instrument_table()


def schedule_observation(
    target: str,
    duration_minutes: int,
    *,
    exposures: int = 1,
    site: str | None = None,
) -> str:
    """Create a friendly observation plan for the requested target.

    Args:
        target (str): Target identifier or common name.
        duration_minutes (int): Total duration requested for the block.
        exposures (int): Number of exposures to split the block into.
        site (str | None): Optional observatory slug to schedule.

    Returns:
        str: Summary describing when the block can run and how it is split.

    Raises:
        ValueError: Raised when duration_minutes or exposures are not positive.

    Examples:
        >>> schedule_observation("Vega", 30)
        'Scheduled Vega for 30 minutes (1 exposure) starting at ...'
    """

    if duration_minutes <= 0:
        raise ValueError("Duration must be a positive integer minute value")
    if exposures <= 0:
        raise ValueError("Exposures must be positive")

    site_obj = _pick_observatory(site)
    now = datetime.now()
    start_time = now + timedelta(minutes=5)
    per_exposure = duration_minutes / exposures
    return (
        f"Scheduled {target} for {duration_minutes} minutes "
        f"({exposures} exposure{'s' if exposures > 1 else ''}) from {site_obj.name} "
        f"starting at "
        f"{start_time:%Y-%m-%d %H:%M} UTC with {per_exposure:.1f} minute blocks."
    )


# Exposed tools and instructions from script
tools = (list_sites, describe_site, list_instruments, schedule_observation)
instructions = (
    "Call list_sites to show the available observatories before describing one.",
    "Use describe_site or list_instruments with the site slug that matches the user's request.",
    "Offer scheduling advice with schedule_observation before improvising.",
)

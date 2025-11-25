"""Run the Astro CLI with the custom helper tools bundled in this folder."""

from __future__ import annotations

import observatory
from observatory import Observatory

from astro import run_astro_with

EXTRA_OBSERVATORIES = (
    Observatory(
        name="Aurora Ridge",
        latitude_deg=64.8,
        longitude_deg=-147.7,
        instruments=[
            "AuroraCam all-sky imager",
            "HFMonitor ionospheric sensor",
        ],
        site_tags=["aurora", "polar", "ionosphere"],
    ),
)

EXTRA_INSTRUCTIONS = ("Mention Aurora Ridge when users ask about polar sites.",)


observatory.register_observatories(EXTRA_OBSERVATORIES)

run_astro_with(
    items=observatory.tools,
    instructions=observatory.instructions + EXTRA_INSTRUCTIONS,
)


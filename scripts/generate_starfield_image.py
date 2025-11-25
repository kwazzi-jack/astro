"""Generate a starfield image for A0 poster."""

import itertools
import math
import random
from collections.abc import Callable

from colour import Color
from PIL import Image, ImageDraw, ImageFont


# Copied color generation and palette from theme.py
def _generate_colour_luminance_variation(base_hex: str, scale: float) -> str:
    """Generate a color variation by scaling luminance within bounds.

    Creates a new color by multiplying the base color's luminance by a scale factor
    and clamping the result between minimum and maximum values.

    Args:
        base_hex (str): Base color in hex format (e.g., "#f5c75f").
        scale (float): Multiplier for luminance adjustment.

    Returns:
        str: Hex color string with adjusted luminance.
    """
    new_color = Color(base_hex)
    new_luminance = new_color.get_luminance() * scale
    new_color.set_luminance(new_luminance)
    return new_color.get_hex_l()


# Core brand
PRIMARY_COLOR = "#d06cc7"
PRIMARY_LIGHT = _generate_colour_luminance_variation(PRIMARY_COLOR, scale=1.2)
PRIMARY_DARK = _generate_colour_luminance_variation(PRIMARY_COLOR, scale=0.5)
PRIMARY_DIM = _generate_colour_luminance_variation(PRIMARY_COLOR, scale=0.2)

SECONDARY_COLOR = "#6c97d0"
SECONDARY_LIGHT = _generate_colour_luminance_variation(SECONDARY_COLOR, scale=1.2)
SECONDARY_DARK = _generate_colour_luminance_variation(SECONDARY_COLOR, scale=0.5)
SECONDARY_DIM = _generate_colour_luminance_variation(SECONDARY_COLOR, scale=0.2)

# Star color map
_pick_star_color = {
    ".": PRIMARY_LIGHT,
    "·": PRIMARY_COLOR,
    "˙": SECONDARY_DARK,
    "•": SECONDARY_LIGHT,
}


# Copied from theme.py
def _make_star_sampler(weights: dict[str, int]) -> Callable[[random.Random], str]:
    """Return a weighted sampler over width-1 star glyphs.

    Args:
        weights (dict[str, int]): Mapping from glyph symbol to relative weight.

    Returns:
        Callable[[random.Random], str]: Callable that produces weighted glyphs
        using the provided random generator.
    """
    weight_items: list[tuple[str, int]] = [
        (symbol, weight) for symbol, weight in weights.items() if weight > 0
    ]
    if not weight_items:
        raise ValueError("No weights provided")

    symbols: tuple[str, ...]
    raw_weights: tuple[int, ...]
    symbols, raw_weights = zip(*weight_items)
    total_weight: float = float(sum(raw_weights))
    cumulative_distribution: list[float] = list(
        itertools.accumulate(w / total_weight for w in raw_weights)
    )

    def pick_star(random_generator: random.Random) -> str:
        r = random_generator.random()
        for i, cum in enumerate(cumulative_distribution):
            if r <= cum:
                return symbols[i]
        return symbols[-1]

    return pick_star


# Weighted width-1 star glyphs
_pick_star = _make_star_sampler({".": 5, "·": 1, "˙": 5, "•": 1})


def _poisson_sample(lam: float, random_generator: random.Random) -> int:
    """Sample from Poisson(lam) with Knuth for small λ, normal approx for large λ.

    Args:
        lam (float): Rate parameter for the Poisson distribution.
        random_generator (random.Random): Random generator used for sampling.

    Returns:
        int: Sampled event count.
    """
    if lam <= 0:
        return 0
    if lam < 30:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random_generator.random()
        return k - 1
    return max(0, int(random_generator.normalvariate(lam, math.sqrt(lam)) + 0.5))


def _generate_starfield_2d_clustered(
    rows: int,
    columns: int,
    base_density: float,  # uniform background probability per cell (e.g., 0.02)
    cluster_rate_per_1000: float,  # expected clusters per 1000 cells (e.g., 1.2)
    cluster_mean_size: float,  # mean stars per cluster (e.g., 8)
    cluster_row_std_cells: float,  # vertical spread in cells (e.g., 2.0)
    cluster_col_std_cells: float,  # horizontal spread in cells (e.g., 4.0)
    random_generator: random.Random,
    pick_star: Callable[[random.Random], str] = _pick_star,
) -> list[list[str]]:
    """Generate a clustered 2D starfield as a grid of glyphs.

    Args:
        rows (int): Number of rows in the generated grid.
        columns (int): Number of columns in the generated grid.
        base_density (float): Uniform background probability per cell.
        cluster_rate_per_1000 (float): Expected clusters per thousand cells.
        cluster_mean_size (float): Mean number of stars per cluster.
        cluster_row_std_cells (float): Vertical spread of each cluster in cells.
        cluster_col_std_cells (float): Horizontal spread of each cluster in
            cells.
        random_generator (random.Random): Random generator used for sampling.
        pick_star (Callable[[random.Random], str]): Glyph sampler used when a
            star should be rendered. Defaults to _pick_star.

    Returns:
        list[list[str]]: Matrix of glyphs representing the starfield.
    """
    # Background layer
    grid: list[list[str]] = [
        [
            "*" if random_generator.random() < base_density else " "
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]

    # Number of clusters ~ Poisson(cluster_rate_per_1000 * (rows * columns / 1000))
    lambda_grid: float = max(0.0, cluster_rate_per_1000 * (rows * columns / 1000.0))
    number_of_clusters: int = _poisson_sample(lambda_grid, random_generator)

    # Place clusters
    for _ in range(number_of_clusters):
        # Cluster center
        center_row = random_generator.uniform(0, rows)
        center_col = random_generator.uniform(0, columns)

        # Cluster size ~ Poisson(cluster_mean_size)
        cluster_size = _poisson_sample(cluster_mean_size, random_generator)

        # Place stars around center
        for _ in range(cluster_size):
            # Gaussian spread
            dr = random_generator.normalvariate(0, cluster_row_std_cells)
            dc = random_generator.normalvariate(0, cluster_col_std_cells)
            r = int(center_row + dr)
            c = int(center_col + dc)
            if 0 <= r < rows and 0 <= c < columns:
                grid[r][c] = pick_star(random_generator)

    # Render marks to weighted glyphs
    for r in range(rows):
        for c in range(columns):
            if grid[r][c] == "*":
                grid[r][c] = pick_star(random_generator)
    return grid


def main():
    # A0 size at 300 DPI
    dpi = 300
    width_mm = 210
    height_mm = 297
    width_px = int(width_mm * dpi / 25.4)
    height_px = int(height_mm * dpi / 25.4)
    print(f"A0 size: {width_px} x {height_px} pixels")

    # Character size approximation
    char_width_px = 10
    char_height_px = 20
    columns = width_px // char_width_px
    rows = height_px // char_height_px
    print(f"Grid: {rows} rows x {columns} columns")

    # Generate starfield
    seed = None  # or 42 for reproducible
    base_density = 0.15
    cluster_rate_per_1000 = 4.0
    cluster_mean_size = 5.0
    cluster_row_std_cells = 1.0
    cluster_col_std_cells = 2.0
    random_generator = random.Random(seed)
    star_grid = _generate_starfield_2d_clustered(
        rows=rows,
        columns=columns,
        base_density=base_density,
        cluster_rate_per_1000=cluster_rate_per_1000,
        cluster_mean_size=cluster_mean_size,
        cluster_row_std_cells=cluster_row_std_cells,
        cluster_col_std_cells=cluster_col_std_cells,
        random_generator=random_generator,
        pick_star=_pick_star,
    )

    # Create image
    img = Image.new(
        "RGBA", (columns * char_width_px, rows * char_height_px), (0, 0, 0, 0)
    )
    draw = ImageDraw.Draw(img)

    # Try to load a monospace font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size=10
        )
    except Exception:
        font = ImageFont.load_default()

    # Draw each character
    for r in range(rows):
        for c in range(columns):
            char = star_grid[r][c]
            if char != " ":
                x = c * char_width_px
                y = r * char_height_px
                color = _pick_star_color.get(
                    char, "#ffffff"
                )  # default to white if not in map
                draw.text((x, y), char, fill=color, font=font)  # white stars

    # Save image
    img.save("starfield_a0.png")
    print("Saved starfield_a0.png")


if __name__ == "__main__":
    main()

from __future__ import annotations

"""Generate synthetic DEM tiles for demonstration purposes."""

from pathlib import Path
import argparse
import numpy as np
import rasterio
from rasterio.transform import from_origin

RAW_DIR = Path("data/raw")


def synthetic_dem(size: int = 100) -> tuple[np.ndarray, float]:
    """Create a DEM with gentle slopes and linear ridges."""
    pixel = 1.0
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    xx, yy = np.meshgrid(x, y)
    slope = 0.1 * (xx + yy)
    dem = slope + np.random.normal(scale=0.05, size=(size, size))

    # Add geometric ridge lines
    dem[size // 3, 10:-10] += 2
    dem[2 * size // 3, 10:-10] += 2
    dem[10:-10, size // 2] += 2
    return dem.astype(np.float32), pixel


def generate_tile(tile_id: str, size: int = 100) -> Path:
    """Generate and save a synthetic DEM tile."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dem, pixel = synthetic_dem(size)
    transform = from_origin(0, size * pixel, pixel, pixel)
    path = RAW_DIR / f"TAL_A01_2018_{tile_id}.grd"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(dem, 1)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic DEM tiles")
    parser.add_argument("tiles", nargs="+", help="Tile IDs to create")
    args = parser.parse_args()
    for tile in args.tiles:
        path = generate_tile(tile)
        print(f"Created {path}")


if __name__ == "__main__":
    main()

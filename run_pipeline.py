"""Batch linear feature detection for priority LiDAR tiles."""
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
import pandas as pd
import geopandas as gpd

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable

from src.utils.detector import detect_lines

# Priority tile IDs
PRIORITY_TILES = [
    "TAL01L0002C0002",
    "TAL01L0001C0003",
    "TAL01L0002C0004",
    "TAL01L0003C0003",
]

RAW_DIR = Path("data/raw")
OUT_DIR = Path("outputs")
REFERENCE = Path("reference/Amazon_geoglyphs.geojson")


def load_dem(tile_id: str) -> tuple[np.ndarray, float]:
    """Load DEM array and pixel size for a given tile."""
    path = RAW_DIR / f"TAL_A01_2018_{tile_id}.grd"
    if not path.exists():
        raise FileNotFoundError(path)

    import rasterio

    with rasterio.open(path) as src:
        data = src.read(1, masked=True).filled(np.nan)
        pixel_size = float(src.res[0])
    return data, pixel_size


def cross_reference(cands: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not REFERENCE.exists():
        cands["status"] = "POSSIBLE_NEW"
        return cands
    ref = gpd.read_file(REFERENCE)
    cands["status"] = "POSSIBLE_NEW"
    intersects = cands.geometry.apply(lambda g: ref.intersects(g).any())
    cands.loc[intersects, "status"] = "KNOWN"
    return cands


def gpt_validate(
    gdf: gpd.GeoDataFrame,
    dem: np.ndarray,
    pixel_size: float,
    out_dir: Path,
    top_n: int = 20,
) -> gpd.GeoDataFrame:
    """Optionally validate candidates with GPT-4 vision."""
    if "OPENAI_API_KEY" not in os.environ:
        gdf["confidence"] = "UNSCORED"
        return gdf

    import base64
    import openai
    from matplotlib import pyplot as plt

    openai.api_key = os.environ["OPENAI_API_KEY"]
    gdf = gdf.copy()
    gdf["confidence"] = "Unscored"
    for idx, row in gdf.head(top_n).iterrows():
        mask = np.zeros_like(dem, dtype=bool)
        coords = np.array(row.geometry.coords)
        rr = (coords[:, 1] / pixel_size).astype(int)
        cc = (coords[:, 0] / pixel_size).astype(int)
        mask[rr.min() : rr.max() + 1, cc.min() : cc.max() + 1] = True
        snippet = np.where(mask, dem, np.nan)

        plt.imshow(snippet, cmap="gray")
        plt.axis("off")
        tmp_path = out_dir / f"snippet_{idx}.png"
        plt.savefig(tmp_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        with open(tmp_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Assess the likelihood that this hillshade shows an archaeological feature. Respond with High, Med, or Low only.",
                            },
                            {
                                "type": "image_url",
                                "image_url": f"data:image/png;base64,{img_b64}",
                            },
                        ],
                    }
                ],
            )
            gdf.loc[idx, "confidence"] = resp.choices[0].message.content.strip()
        except Exception:
            gdf.loc[idx, "confidence"] = "API_ERROR"
    return gdf


def build_interactive_map() -> None:
    """Create Folium map linking candidate GeoJSONs."""
    import folium

    m = folium.Map(location=[-3.0, -60.0], zoom_start=5)
    for tile in PRIORITY_TILES:
        path = OUT_DIR / tile / "candidates.geojson"
        if not path.exists():
            continue
        gdf = gpd.read_file(path)
        folium.GeoJson(gdf, name=tile).add_to(m)
    m.save(OUT_DIR / "interactive_map.html")


def process_tile(tile_id: str) -> None:
    out_dir = OUT_DIR / tile_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dem, pixel_size = load_dem(tile_id)

    lines = detect_lines(
        dem,
        pixel_size=pixel_size,
        min_length=15.0,
        max_length=450.0,
        max_gap=25.0,
    )
    gdf = gpd.GeoDataFrame(geometry=lines, crs="EPSG:4326")
    gdf = cross_reference(gdf)
    gdf = gpt_validate(gdf, dem, pixel_size, out_dir)
    gdf.to_file(out_dir / "candidates.geojson", driver="GeoJSON")

    summary = pd.DataFrame(
        {
            "tile": tile_id,
            "line_id": range(len(lines)),
            "length_m": [ln.length for ln in lines],
            "bearing_deg": [
                (
                    np.degrees(
                        np.arctan2(
                            ln.coords[-1][1] - ln.coords[0][1],
                            ln.coords[-1][0] - ln.coords[0][0],
                        )
                    )
                    % 360
                )
                for ln in lines
            ],
            "centroid_lon": [ln.centroid.x for ln in lines],
            "centroid_lat": [ln.centroid.y for ln in lines],
            "status": gdf["status"],
            "confidence": gdf["confidence"],
        }
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    from matplotlib import pyplot as plt
    from matplotlib.colors import LightSource

    for az in (45, 135, 225, 315):
        ls = LightSource(azdeg=az, altdeg=45)
        hs = ls.hillshade(dem, vert_exag=1, dx=pixel_size, dy=pixel_size)
        path = out_dir / f"hillshade_{az}.png"

        plt.imshow(hs, cmap="gray")
        plt.axis("off")
        plt.savefig(path, bbox_inches="tight", pad_inches=0)
        plt.close()


def main(tiles: list[str] | None = None) -> None:
    """Run the pipeline for a list of tile IDs."""
    OUT_DIR.mkdir(exist_ok=True)
    tile_list = tiles or PRIORITY_TILES
    for tile in tqdm(tile_list, desc="Tiles"):
        try:
            process_tile(tile)
        except FileNotFoundError:
            print(f"Tile {tile} not found, skipping")
    build_interactive_map()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process LiDAR tiles")
    parser.add_argument("--tiles", nargs="*", help="Tile IDs to process")
    args = parser.parse_args()
    main(args.tiles)

import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

import run_pipeline as rp
from shapely.geometry import LineString


def test_load_dem(tmp_path, monkeypatch):
    arr = np.arange(9, dtype=np.float32).reshape(3, 3)
    transform = from_origin(0, 3, 1.0, 1.0)
    path = tmp_path / "TAL_A01_2018_TEST.grd"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=3,
        width=3,
        count=1,
        dtype="float32",
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(arr, 1)

    monkeypatch.setattr(rp, "RAW_DIR", tmp_path)
    data, pixel = rp.load_dem("TEST")
    assert np.allclose(data, arr)
    assert pixel == 1.0


def test_gpt_validate_no_key(tmp_path, monkeypatch):
    gdf = rp.gpd.GeoDataFrame({"geometry": [LineString([(0, 0), (1, 0)])]})
    gdf["status"] = ["POSSIBLE_NEW"]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = rp.gpt_validate(gdf, np.zeros((2, 2)), 1.0, tmp_path)
    assert (out["confidence"] == "UNSCORED").all()

def test_process_tile(tmp_path, monkeypatch):
    arr = np.zeros((5, 5), dtype=np.float32)
    transform = from_origin(0, 5, 1, 1)
    raster = tmp_path / "TAL_A01_2018_X.grd"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        transform=transform,
        crs="EPSG:4326",
    ) as dst:
        dst.write(arr, 1)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(rp, "RAW_DIR", tmp_path)
    monkeypatch.setattr(rp, "OUT_DIR", out_dir)
    monkeypatch.setattr(rp, "REFERENCE", tmp_path / "ref.geojson")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    rp.process_tile("X")
    assert (out_dir / "X" / "candidates.geojson").exists()


def test_main_with_tiles(tmp_path, monkeypatch):
    arr = np.zeros((5, 5), dtype=np.float32)
    raster = tmp_path / "TAL_A01_2018_Y.grd"
    with rasterio.open(
        raster,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        transform=from_origin(0, 5, 1, 1),
        crs="EPSG:4326",
    ) as dst:
        dst.write(arr, 1)

    monkeypatch.setattr(rp, "RAW_DIR", tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(rp, "OUT_DIR", out_dir)
    monkeypatch.setattr(rp, "REFERENCE", tmp_path / "ref.geojson")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    rp.main(["Y"])
    assert (out_dir / "Y" / "summary.csv").exists()


def test_sample_data_generator(tmp_path):
    import sample_data

    monkeypatch_dir = tmp_path / "data" / "raw"
    monkeypatch_dir.mkdir(parents=True)
    sample_data.RAW_DIR = monkeypatch_dir
    path = sample_data.generate_tile("Z", size=10)
    assert path.exists()
    with rasterio.open(path) as src:
        data = src.read(1)
    assert data.shape == (10, 10)

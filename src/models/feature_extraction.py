#!/usr/bin/env python
"""
Amazon Deep Insights - Feature Extraction Module
================================================

Utility functions for extracting information from raster layers
derived from LiDAR data (e.g. CHM, DEM) such as:

* Tree-top detection from Canopy-Height-Models (CHM)
* Terrain derivatives – slope, aspect, curvature
* Simple archaeological feature detection (candidate mounds / depressions)

The functions are *lightweight* and rely only on commonly available
scientific–python libraries.  They are intended for **prototype** use-cases
and can be replaced with more sophisticated domain-specific algorithms later.
"""

from __future__ import annotations

from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import measure, morphology

# Optional, load lazily to avoid hard dependency when not required
try:
    import rasterio
    from rasterio.transform import Affine
except ImportError:  # pragma: no cover
    rasterio = None  # type: ignore
    Affine = object  # type: ignore

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon
except ImportError:  # pragma: no cover
    gpd = None  # type: ignore
    Point = Polygon = object  # type: ignore

###############################################################################
# Helper utilities
###############################################################################


def _array_index_to_coords(
    rows: np.ndarray,
    cols: np.ndarray,
    transform: "Affine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert array row / col indices to map x / y using a rasterio Affine.
    """

    if not isinstance(transform, Affine):
        raise TypeError("transform must be rasterio.transform.Affine")

    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    return np.array(xs), np.array(ys)


###############################################################################
# Tree detection
###############################################################################


def detect_tree_tops(
    chm: np.ndarray,
    transform: "Affine",
    *,
    min_height: float = 5.0,
    min_distance: int = 5,
    gaussian_sigma: float = 1.0,
) -> "gpd.GeoDataFrame":
    """
    Detect tree-tops (local maxima) in a Canopy Height Model (CHM).

    Parameters
    ----------
    chm : np.ndarray
        2-D array of canopy heights.
    transform : Affine
        Georeferencing transform for the CHM.
    min_height : float, default 5.0
        Minimum height (metres) for a pixel to be considered in the search.
    min_distance : int, default 5
        Minimum number of pixels separating detected peaks.
    gaussian_sigma : float, default 1.0
        Standard deviation for pre-smoothing.  0 disables smoothing.

    Returns
    -------
    GeoDataFrame with columns:
        ["x", "y", "height"]
    """

    if chm.ndim != 2:
        raise ValueError("CHM must be 2-D")

    # Mask low canopy
    chm_masked = np.where(chm >= min_height, chm, 0)

    if gaussian_sigma > 0:
        chm_filtered = ndi.gaussian_filter(chm_masked, gaussian_sigma)
    else:
        chm_filtered = chm_masked

    # Detect peaks
    coords = peak_local_max(
        chm_filtered,
        min_distance=min_distance,
        threshold_abs=min_height,
        indices=True,  # type: ignore[arg-type]
    )

    if coords.size == 0:
        # Return empty GeoDataFrame
        return gpd.GeoDataFrame(columns=["x", "y", "height"], geometry=[], crs="EPSG:4326")  # type: ignore[arg-type]

    rows, cols = coords[:, 0], coords[:, 1]
    xs, ys = _array_index_to_coords(rows, cols, transform)
    heights = chm[rows, cols]

    df = pd.DataFrame({"x": xs, "y": ys, "height": heights})
    gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(xs, ys)], crs="EPSG:4326")  # type: ignore[arg-type]
    return gdf


###############################################################################
# Terrain derivatives
###############################################################################


def compute_slope_aspect(
    dem: np.ndarray,
    transform: "Affine",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute slope and aspect in *degrees* from a Digital Elevation Model.

    Returns
    -------
    slope : np.ndarray
    aspect : np.ndarray   (0–360°, 0 = North)
    """

    # Cellsize
    dx = transform.a
    dy = -transform.e  # negative because affine e is -pixel height

    # Gradient
    dz_dx = ndi.sobel(dem, axis=1, mode="nearest") / (8 * dx)
    dz_dy = ndi.sobel(dem, axis=0, mode="nearest") / (8 * dy)

    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    aspect = np.degrees(np.arctan2(dz_dx, dz_dy))
    aspect = np.where(aspect < 0, 360.0 + aspect, aspect)

    return slope, aspect


def curvature_metrics(
    dem: np.ndarray,
    transform: "Affine",
) -> Dict[str, np.ndarray]:
    """
    Calculate plan and profile curvature using finite differences.

    Formulas adapted from Evans (1980).  Cellsize assumed square.
    """

    cellsize = transform.a

    # First derivatives
    dz_dx = ndi.sobel(dem, axis=1, mode="nearest") / (8 * cellsize)
    dz_dy = ndi.sobel(dem, axis=0, mode="nearest") / (8 * cellsize)

    # Second derivatives
    d2z_dx2 = ndi.sobel(dz_dx, axis=1, mode="nearest") / (8 * cellsize)
    d2z_dy2 = ndi.sobel(dz_dy, axis=0, mode="nearest") / (8 * cellsize)
    d2z_dxdy = ndi.sobel(dz_dx, axis=0, mode="nearest") / (8 * cellsize)

    p_curv = (
        (dz_dx**2 * d2z_dx2 + 2 * dz_dx * dz_dy * d2z_dxdy + dz_dy**2 * d2z_dy2)
        / ((dz_dx**2 + dz_dy**2) * np.sqrt(dz_dx**2 + dz_dy**2) + 1e-12)
    )

    plan_curv = (
        (dz_dy**2 * d2z_dx2 - 2 * dz_dx * dz_dy * d2z_dxdy + dz_dx**2 * d2z_dy2)
        / ((dz_dx**2 + dz_dy**2) ** 1.5 + 1e-12)
    )

    return {"profile": p_curv, "plan": plan_curv}


###############################################################################
# Archaeological feature detection (prototype)
###############################################################################


def detect_mound_features(
    dem: np.ndarray,
    transform: "Affine",
    *,
    gaussian_size: int = 51,
    height_threshold: float = 1.0,
    area_threshold: float = 200.0,
) -> "gpd.GeoDataFrame":
    """
    Simple candidate mound / earthwork detector.

    Steps
    -----
    1. Local-relief model (DEM - Gaussian-smoothed DEM)
    2. Threshold positive residuals > `height_threshold`
    3. Remove small components (< `area_threshold` m²)
    4. Convert blobs to polygons returned as GeoDataFrame
    """

    # Build Local Relief Model (LRM)
    smooth = ndi.gaussian_filter(dem, gaussian_size / 6)  # sigma ≈ size/6
    residual = dem - smooth

    # Threshold
    mask = residual > height_threshold

    # Label connected components
    labeled = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled)

    polygons: List[Polygon] = []
    heights: List[float] = []

    # Cell area (m²)
    cell_area = transform.a * abs(transform.e)

    for prop in props:
        area_m2 = prop.area * cell_area
        if area_m2 < area_threshold:
            continue

        # Convert pixel coords to world coords
        rows, cols = np.where(labeled == prop.label)
        xs, ys = _array_index_to_coords(rows, cols, transform)
        # Build polygon (convex hull of points)
        poly = Polygon(zip(xs, ys)).convex_hull
        polygons.append(poly)
        heights.append(residual[rows, cols].max())

    gdf = gpd.GeoDataFrame(
        {"height": heights, "area_m2": [p.area for p in polygons]},
        geometry=polygons,
        crs="EPSG:4326",
    )

    return gdf


__all__ = [
    "detect_tree_tops",
    "compute_slope_aspect",
    "curvature_metrics",
    "detect_mound_features",
]


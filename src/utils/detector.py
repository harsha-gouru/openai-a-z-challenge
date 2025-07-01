"""Simple linear feature detector for DEM arrays."""
from __future__ import annotations

from typing import Iterable, List
import numpy as np
from shapely.geometry import LineString
from scipy import ndimage as ndi

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(iterable: Iterable, **kwargs):  # type: ignore
        return iterable


def _edges(dem: np.ndarray) -> np.ndarray:
    """Compute simple edge mask from DEM using gradient magnitude."""
    grad_y, grad_x = np.gradient(dem)
    magnitude = np.hypot(grad_x, grad_y)
    threshold = np.percentile(magnitude, 95)
    return magnitude > threshold


def detect_lines(
    dem: np.ndarray,
    *,
    pixel_size: float,
    min_length: float = 15.0,
    max_length: float = 450.0,
    max_gap: float = 10.0,
) -> List[LineString]:
    """Detect linear features as simple connected edge components."""
    if dem.ndim != 2:
        raise ValueError("DEM must be 2-D")

    edge_mask = _edges(dem)
    gap_px = max(1, int(max_gap / pixel_size))
    closed = ndi.binary_closing(edge_mask, iterations=gap_px)
    labeled, num = ndi.label(closed)

    lines: List[LineString] = []
    for label in range(1, num + 1):
        rows, cols = np.where(labeled == label)
        if rows.size < 2:
            continue
        length = np.hypot(rows.max() - rows.min(), cols.max() - cols.min()) * pixel_size
        if length < min_length or length > max_length:
            continue
        line = LineString(
            [
                (cols.min() * pixel_size, rows.min() * pixel_size),
                (cols.max() * pixel_size, rows.max() * pixel_size),
            ]
        )
        lines.append(line)
    return lines


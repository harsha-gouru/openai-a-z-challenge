#!/usr/bin/env python
"""
Amazon Deep Insights – PDAL/GDAL pipeline helpers
=================================================

This module contains **pure-python helpers** that *return* ready-to-run
`dict` definitions for [`pdal.Pipeline`](https://pdal.io/) as well as a few
convenience wrappers for common raster-generation steps carried out with
*GDAL*.

Keeping the logic here (separate from the heavier `lidar_processing.py`) allows
other parts of the code-base – notebooks, Streamlit callbacks, unit-tests – to
instantiate *declarative* pipelines without re-implementing boiler-plate.

Typical Usage
-------------
```python
from pdal import Pipeline
from src.preprocessing.pipelines import make_dem_pipeline, run_pipeline

pipe_def = make_dem_pipeline("tile.laz", "tile_dem_1m.tif", resolution=1.0)
run_pipeline(pipe_def)        # executes & returns pdal.Pipeline object
```
"""

from __future__ import annotations

import json
from typing import Dict, Any, Optional

import logging

try:
    import pdal  # heavy import – only required when actually executing
except ImportError:  # pragma: no cover
    pdal = None  # type: ignore

# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #


def run_pipeline(pipeline_def: Dict[str, Any]) -> "pdal.Pipeline":  # type: ignore[name-defined]
    """
    Execute a PDAL pipeline *definition* and return the executed Pipeline
    instance.

    This tiny wrapper avoids importing `pdal` at call-sites that only need the
    JSON spec.
    """

    if pdal is None:  # pragma: no cover
        raise ImportError("pdal is not installed – cannot run pipelines")

    pipeline = pdal.Pipeline(json.dumps(pipeline_def))
    logging.getLogger(__name__).debug("Executing PDAL pipeline …")
    pipeline.execute()
    return pipeline


# --------------------------------------------------------------------------- #
# Pipeline factory functions
# --------------------------------------------------------------------------- #

def make_denoise_pipeline(
    input_file: str,
    output_file: str,
    *,
    mean_k: int = 8,
    multiplier: float = 3.0,
) -> Dict[str, Any]:
    """
    Basic statistical outlier removal.

    Parameters
    ----------
    mean_k : int
        Number of neighbours.
    multiplier : float
        Distance multiplier; points whose mean distance to neighbours is larger
        than *mean distance × multiplier* are removed.
    """

    return {
        "pipeline": [
            input_file,
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": mean_k,
                "multiplier": multiplier,
            },
            {
                "type": "writers.las",
                "filename": output_file,
            },
        ]
    }


def make_dem_pipeline(
    input_file: str,
    output_file: str,
    *,
    resolution: float = 1.0,
    interpolation: str = "idw",
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Create a *Digital Elevation Model* pipeline (ground-only).
    """

    return {
        "pipeline": [
            input_file,
            {"type": "filters.assign", "assignment": "Classification[:]=0"},
            {"type": "filters.elm"},  # elevation-based ground filter
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 8,
                "multiplier": 3.0,
            },
            {"type": "filters.pmf"},  # progressive morphological filter
            {"type": "filters.range", "limits": "Classification[2:2]"},  # keep ground
            {
                "type": "writers.gdal",
                "filename": output_file,
                "output_type": interpolation,
                "resolution": resolution,
                "window_size": window_size,
            },
        ]
    }


def make_dsm_pipeline(
    input_file: str,
    output_file: str,
    *,
    resolution: float = 1.0,
    window_size: int = 10,
) -> Dict[str, Any]:
    """
    Create a *Digital Surface Model* pipeline (max ‑ first return).
    """

    return {
        "pipeline": [
            input_file,
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 8,
                "multiplier": 3.0,
            },
            {
                "type": "writers.gdal",
                "filename": output_file,
                "output_type": "max",
                "resolution": resolution,
                "window_size": window_size,
            },
        ]
    }


# --------------------------------------------------------------------------- #
# CHM helper (GDAL calc wrapper)
# --------------------------------------------------------------------------- #

def make_chm_pipeline(
    dem_file: str,
    dsm_file: str,
    chm_file: str,
) -> Dict[str, Any]:
    """
    Return a **PDAL pipeline with a single GDAL `calc`** stage that subtracts
    the DEM from the DSM and writes a *Canopy Height Model*.

    PDAL 2.6+ contains a `filters.gdal` that can do basic raster math.
    """

    return {
        "pipeline": [
            {
                "type": "filters.gdal",
                "dimension": "Z",
                "raster": dsm_file,
            },
            {
                "type": "filters.gdal",
                "dimension": "Z",
                "raster": dem_file,
                "gdaloptions": {
                    "calc": "A-B",
                },
            },
            {
                "type": "writers.gdal",
                "filename": chm_file,
                "output_type": "mean",
                "resolution": 1.0,
            },
        ]
    }


# --------------------------------------------------------------------------- #
# Public exports
# --------------------------------------------------------------------------- #

__all__ = [
    "make_denoise_pipeline",
    "make_dem_pipeline",
    "make_dsm_pipeline",
    "make_chm_pipeline",
    "run_pipeline",
]


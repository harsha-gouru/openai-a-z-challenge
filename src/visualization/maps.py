#!/usr/bin/env python
"""
Amazon Deep Insights – Mapping helpers
=====================================

Utility wrappers around *folium* (Leaflet.js) so notebooks and the Streamlit
front-end can quickly add common layers:

* Basemap initialisation (`create_base_map`)
* PNG overlay for single-band rasters (e.g. DEM / CHM)
* Points & polygons from GeoPandas
* Simple heat-maps for point intensity

The functions are **lightweight** – they keep external dependencies minimal
and avoid heavy GIS servers.  For web-scale visualisation consider tiling
rasters with *rio-cogeo* / *titiler*.
"""

from __future__ import annotations

from typing import Iterable, List, Callable, Dict, Any
import base64
import io
import warnings

import numpy as np

import folium
from folium import plugins

# Optional dependencies – imported lazily
try:  # Raster display
    import rasterio
    from rasterio.enums import Resampling
except ImportError:  # pragma: no cover
    rasterio = None  # type: ignore

try:  # GeoDataFrame styling
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None  # type: ignore

try:  # Colour maps
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ImportError:  # pragma: no cover
    plt = None  # type: ignore
    mpl = None  # type: ignore

###############################################################################
# Basemap helpers
###############################################################################


def create_base_map(
    center: Iterable[float] | None = None,
    *,
    zoom: int = 5,
    tiles: str = "CartoDB positron",
) -> folium.Map:
    """
    Create a Leaflet basemap with fullscreen, layer control & measure widgets.

    Parameters
    ----------
    center
        Lat / lon pair.  Defaults to approximate centre of Amazon.
    zoom
        Initial zoom level.
    tiles
        Basemap tiles – any valid folium tile string or URL template.
    """

    if center is None:
        center = (-3.4653, -62.2159)

    m = folium.Map(location=center, zoom_start=zoom, tiles=tiles, control_scale=True)

    # UX plugins
    plugins.Fullscreen().add_to(m)
    plugins.MeasureControl(position="bottomleft", primary_length_unit="meters").add_to(m)
    folium.LayerControl(collapsed=True).add_to(m)
    return m


###############################################################################
# Raster overlay
###############################################################################


def _auto_cmap(
    arr: np.ndarray,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """
    Convert a single-band float array → uint8 RGBA using Matplotlib colormap.
    """

    if mpl is None:  # pragma: no cover
        raise ImportError("matplotlib required for raster colouring")

    if vmin is None:
        vmin = np.nanpercentile(arr, 2)
    if vmax is None:
        vmax = np.nanpercentile(arr, 98)

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_obj = mpl.cm.get_cmap(cmap)
    rgba = cmap_obj(norm(arr, copy=False), bytes=True)  # type: ignore[arg-type]
    return rgba


def add_raster_overlay(
    m: folium.Map,
    raster_path: str,
    *,
    name: str | None = None,
    cmap: str = "viridis",
    opacity: float = 0.6,
    max_size: int = 2048,
) -> None:
    """
    Overlay a small/medium **single-band raster** onto a folium map as an
    inline PNG image (base64).  Intended for quick-look visualisation – not
    suitable for huge rasters.
    """

    if rasterio is None:  # pragma: no cover
        raise ImportError("rasterio required for add_raster_overlay")

    if name is None:
        name = raster_path.split("/")[-1]

    with rasterio.open(raster_path) as src:
        # Determine output shape (preserve aspect ratio)
        scale = max(src.width / max_size, src.height / max_size, 1)
        out_width = int(src.width / scale)
        out_height = int(src.height / scale)

        arr = src.read(
            1,
            out_shape=(1, out_height, out_width),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype(np.float32)

        rgba = _auto_cmap(arr, cmap=cmap)

        # Encode as PNG in-memory
        img = io.BytesIO()
        if plt is None:  # pragma: no cover
            raise ImportError("matplotlib required for PNG encoding")
        plt.imsave(img, rgba, format="png")
        img64 = base64.b64encode(img.getvalue()).decode("utf-8")

        bounds = [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]

    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{img64}",
        bounds=bounds,
        opacity=opacity,
        name=name,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)


###############################################################################
# Vector layers
###############################################################################


def add_point_features(
    m: folium.Map,
    gdf: "gpd.GeoDataFrame",
    *,
    popup_fields: List[str] | None = None,
    marker_color: str = "green",
    marker_icon: str = "info-sign",
    name: str = "Points",
) -> None:
    """
    Add point features from a GeoDataFrame to the map.
    """

    if gpd is None:  # pragma: no cover
        raise ImportError("GeoPandas required for point visualisation")

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom.is_empty or geom.geom_type != "Point":
            continue

        popup_html = "<br>".join([f"<b>{col}</b>: {row[col]}" for col in popup_fields or []])
        folium.Marker(
            location=[geom.y, geom.x],
            popup=popup_html if popup_html else None,
            icon=folium.Icon(color=marker_color, icon=marker_icon),
        ).add_to(m)

    folium.LayerControl().add_to(m)


def add_polygon_features(
    m: folium.Map,
    gdf: "gpd.GeoDataFrame",
    *,
    style_fn: Callable[[Dict[str, Any]], Dict[str, Any]] | None = None,
    name: str = "Polygons",
) -> None:
    """
    Add polygon/line features via folium.GeoJson.
    """

    if gpd is None:  # pragma: no cover
        raise ImportError("GeoPandas required for polygon visualisation")

    style_fn = style_fn or (lambda _: {"color": "#FF7800", "weight": 2, "fillOpacity": 0.3})

    folium.GeoJson(
        data=gdf.to_json(),
        style_function=style_fn,
        name=name,
    ).add_to(m)


###############################################################################
# Heat-map
###############################################################################


def add_heatmap(
    m: folium.Map,
    gdf: "gpd.GeoDataFrame",
    *,
    value_col: str | None = None,
    radius: int = 8,
    blur: int = 15,
    max_zoom: int = 12,
    name: str = "Heatmap",
) -> None:
    """
    Add a heat-map layer based on point density / magnitude.
    """

    if gpd is None:  # pragma: no cover
        raise ImportError("GeoPandas required for heatmap visualisation")

    if gdf.empty:
        warnings.warn("GeoDataFrame is empty – heatmap skipped")
        return

    pts = []
    for _, row in gdf.iterrows():
        if row.geometry.is_empty or row.geometry.geom_type != "Point":
            continue
        lat, lon = row.geometry.y, row.geometry.x
        val = row[value_col] if value_col else 1
        pts.append([lat, lon, val])

    plugins.HeatMap(pts, radius=radius, blur=blur, max_zoom=max_zoom, name=name).add_to(m)


__all__ = [
    "create_base_map",
    "add_raster_overlay",
    "add_point_features",
    "add_polygon_features",
    "add_heatmap",
]


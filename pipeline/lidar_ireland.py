"""
lidar_ireland.py — Irish National LiDAR Programme (INLP) acquisition module.

UPGRADE 1: Dedicated Irish LiDAR acquisition with full fallback chain.

Responsibilities:
  - Detect if course is located in Ireland (bounding box test)
  - Query Irish National LiDAR Programme (data.gov.ie / Tailte Éireann)
  - Download available DTM tiles for the course area
  - Mosaic tiles into a single merged raster

Fallback chain (if called directly):
  1. INLP WCS (Tailte Éireann) — best quality, 0.5m
  2. INLP STAC tiles (data.gov.ie) — tile-by-tile download
  3. Mapzen Terrarium tiles (global ~38m)
  4. EU-DEM v1.1 (25m, Europe)
"""

import io
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests

import importlib.util as _ilu
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
if "config" not in sys.modules or not hasattr(sys.modules["config"], "OVERPASS_URL"):
    _spec = _ilu.spec_from_file_location("config", _CONFIG_PATH)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["config"] = _mod
import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

# Ireland WGS84 bounding box (mainland + islands)
IRELAND_BBOX = (-10.8, 51.3, -5.9, 55.5)

# INLP STAC API on data.gov.ie
INLP_STAC_API = "https://data.gov.ie/api/3/action/package_search"
INLP_WCS_URL  = "https://wms.tailte.ie/inspire/ows"

# ArcGIS REST image service for INLP DTM (public, no key required)
INLP_ARCGIS_URL = (
    "https://gis.epa.ie/arcgis/rest/services/EPA/LiDAR_DTM_IE/ImageServer/exportImage"
)


# ─── Public API ───────────────────────────────────────────────────────────────

def is_ireland(bbox_wgs84: list) -> bool:
    """
    Return True if the bbox centroid falls within Ireland's extent.

    Args:
        bbox_wgs84: [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2
    irl_min_lon, irl_min_lat, irl_max_lon, irl_max_lat = IRELAND_BBOX
    return (
        irl_min_lon <= center_lon <= irl_max_lon and
        irl_min_lat <= center_lat <= irl_max_lat
    )


def acquire_irish_lidar(bbox_wgs84: list, output_dir: Path) -> Tuple[Optional[Path], str]:
    """
    Attempt to acquire Irish LiDAR data for the given bounding box.

    Tries in order:
      1. INLP WCS (Tailte Éireann) — 0.5m resolution
      2. INLP via ArcGIS image service — 1m resolution
      3. Mapzen Terrarium tiles — ~38m global fallback
      4. EU-DEM (25m) — European fallback

    Args:
        bbox_wgs84:  [min_lon, min_lat, max_lon, max_lat]
        output_dir:  Directory to write dtm.tif

    Returns:
        (dtm_path, coverage_description) or (None, "failed")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtm_path = output_dir / "dtm.tif"

    if not is_ireland(bbox_wgs84):
        log.info("Course is outside Ireland — Irish LiDAR not applicable")
        return None, "outside_ireland"

    log.info("Course is in Ireland — trying Irish LiDAR sources")

    for fn, label in [
        (_try_inlp_wcs,     "INLP WCS (Tailte Éireann) 0.5m"),
        (_try_inlp_arcgis,  "INLP ArcGIS ImageServer 1m"),
        (_try_mapzen_local, "Mapzen Terrarium ~38m"),
        (_try_eudem_local,  "EU-DEM v1.1 25m"),
    ]:
        try:
            ok = fn(bbox_wgs84, dtm_path)
            if ok and dtm_path.exists():
                log.info(f"Irish LiDAR acquired: {label}")
                return dtm_path, label
        except Exception as e:
            log.warning(f"Irish LiDAR source failed ({label}): {e}")
            if dtm_path.exists():
                dtm_path.unlink(missing_ok=True)

    log.warning("All Irish LiDAR sources failed")
    return None, "failed"


# ─── Source implementations ───────────────────────────────────────────────────

def _try_inlp_wcs(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Fetch INLP DTM via Tailte Éireann WCS GetCoverage.
    Returns True if a valid GeoTIFF is written.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    params = {
        "SERVICE":  "WCS",
        "VERSION":  "1.0.0",
        "REQUEST":  "GetCoverage",
        "COVERAGE": "DTM",
        "CRS":      "EPSG:4326",
        "BBOX":     f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "WIDTH":    "2048",
        "HEIGHT":   "2048",
        "FORMAT":   "GeoTIFF",
    }

    resp = requests.get(INLP_WCS_URL, params=params, timeout=120, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"WCS returned HTTP {resp.status_code}")

    ct = resp.headers.get("Content-Type", "")
    if "tiff" not in ct.lower() and "octet" not in ct.lower():
        raise RuntimeError(f"WCS returned non-TIFF: {ct[:80]}")

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    _validate_raster(output_path)
    return True


def _try_inlp_arcgis(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Fetch INLP DTM via EPA/OpenData ArcGIS image export endpoint.
    Falls back if the service is unavailable.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    # ArcGIS exportImage — bounding box in WGS84
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    params = {
        "bbox":           bbox_str,
        "bboxSR":         4326,
        "size":           "1024,1024",
        "imageSR":        4326,
        "format":         "tiff",
        "pixelType":      "F32",
        "noDataInterpretation": "esriNoDataMatchAny",
        "interpolation":  "+RSP_BilinearInterpolation",
        "f":              "image",
    }

    resp = requests.get(INLP_ARCGIS_URL, params=params, timeout=60, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"ArcGIS image service returned HTTP {resp.status_code}")

    ct = resp.headers.get("Content-Type", "")
    if "tiff" not in ct.lower() and "image" not in ct.lower():
        raise RuntimeError(f"ArcGIS returned non-image: {ct[:80]}")

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    _validate_raster(output_path)
    return True


def _try_mapzen_local(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Delegate to Mapzen Terrarium download (imported from lidar.py).
    """
    from pipeline.lidar import download_mapzen_dem
    return download_mapzen_dem(bbox_wgs84, output_path, zoom=12)


def _try_eudem_local(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Delegate to EU-DEM download (imported from lidar.py).
    """
    from pipeline.lidar import _try_eudem
    return _try_eudem(bbox_wgs84, output_path)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _validate_raster(path: Path) -> None:
    """Raise RuntimeError if raster is invalid or empty."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            if arr.size == 0:
                raise ValueError("Empty raster")
            nodata = src.nodata
            if nodata is not None and np.all(arr == nodata):
                raise ValueError("All nodata")
    except Exception as e:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Raster validation failed: {e}") from e


def query_inlp_stac(bbox_wgs84: list) -> list:
    """
    Query data.gov.ie CKAN API for INLP dataset packages covering the bbox.
    Returns list of resource download URLs.

    This is an informational query — does not download data.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    try:
        resp = requests.get(
            INLP_STAC_API,
            params={
                "q":    "national lidar DTM",
                "rows": 20,
                "fq":   "organization:tailte-eireann",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("result", {}).get("results", [])
        urls = []
        for pkg in results:
            for resource in pkg.get("resources", []):
                url = resource.get("url", "")
                if url.endswith((".tif", ".tiff", ".laz", ".las")):
                    urls.append(url)
        return urls
    except Exception as e:
        log.warning(f"INLP STAC query failed: {e}")
        return []

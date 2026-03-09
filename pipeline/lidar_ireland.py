"""
lidar_ireland.py — Irish National LiDAR Programme (INLP) acquisition.

Acquisition chain for Irish courses:
  1. Tailte Éireann WCS — 0.5m DTM (best quality)
  2. EPA/OpenData ArcGIS ImageServer — 1m DTM
  3. data.gov.ie STAC catalogue query (informational; tiles require auth)
  4. Delegate to lidar.py Mapzen fallback

Only attempted when the course centroid falls within Ireland's extent.
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
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py"
)
if "config" not in sys.modules or not hasattr(sys.modules["config"], "OVERPASS_URL"):
    _spec = _ilu.spec_from_file_location("config", _CONFIG_PATH)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["config"] = _mod
import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

# WGS84 bounding box for Ireland (mainland + islands)
IRELAND_BBOX = (-10.8, 51.3, -5.9, 55.5)

# Tailte Éireann WCS — OGC Web Coverage Service
_INLP_WCS_URL = getattr(config, "INLP_WCS_URL",
                         "https://wms.tailte.ie/inspire/ows")

# EPA/OpenData ArcGIS ImageServer — public, no auth required
_INLP_ARCGIS_ENDPOINTS = [
    # Primary — EPA LiDAR DTM
    getattr(config, "INLP_ARCGIS_URL",
            "https://gis.epa.ie/arcgis/rest/services/EPA/LiDAR_DTM_IE/ImageServer/exportImage"),
    # Alternate — national mapping via ArcGIS Online
    "https://services.arcgisonline.com/arcgis/rest/services/Elevation/World_Hillshade/ImageServer/exportImage",
]

# INLP STAC / CKAN API
_INLP_STAC_API = getattr(config, "INLP_STAC_URL",
                          "https://data.gov.ie/api/3/action/package_search")


# ─── Public API ───────────────────────────────────────────────────────────────

def is_ireland(bbox_wgs84: list) -> bool:
    """Return True if the bbox centroid lies within Ireland's WGS84 extent."""
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    cx = (min_lon + max_lon) / 2
    cy = (min_lat + max_lat) / 2
    irl_w, irl_s, irl_e, irl_n = IRELAND_BBOX
    return irl_w <= cx <= irl_e and irl_s <= cy <= irl_n


def acquire_irish_lidar(
    bbox_wgs84: list,
    output_dir: Path,
) -> Tuple[Optional[Path], str]:
    """
    Acquire Irish LiDAR data for the given bounding box.

    Tries in order:
      1. INLP WCS (Tailte Éireann) — 0.5m
      2. INLP ArcGIS ImageServer   — 1m
      3. Mapzen Terrarium           — ~38m global fallback

    Returns:
        (dtm_path, coverage_label) or (None, "failed")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtm_path = output_dir / "dtm.tif"

    if not is_ireland(bbox_wgs84):
        log.info("  Course outside Ireland — Irish LiDAR not applicable")
        return None, "outside_ireland"

    log.info("  Course is in Ireland — trying INLP sources")

    sources = [
        (_try_inlp_wcs,    "Tailte Éireann WCS 0.5m"),
        (_try_inlp_arcgis, "EPA ArcGIS ImageServer 1m"),
        (_try_mapzen_local,"Mapzen Terrarium ~38m"),
    ]

    for fn, label in sources:
        try:
            ok = fn(bbox_wgs84, dtm_path)
            if ok and dtm_path.exists() and dtm_path.stat().st_size > 5000:
                _validate_raster(dtm_path)
                log.info(f"  Irish LiDAR acquired: {label}")
                return dtm_path, label
        except Exception as e:
            log.warning(f"  INLP source failed ({label}): {e}")
            dtm_path.unlink(missing_ok=True)

    log.warning("  All Irish LiDAR sources failed")
    return None, "failed"


# ─── Source implementations ───────────────────────────────────────────────────

def _try_inlp_wcs(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Fetch INLP DTM via OGC WCS 1.0.0 GetCoverage from Tailte Éireann.

    The service requires the bbox in EPSG:4326 and returns a GeoTIFF.
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

    resp = requests.get(
        _INLP_WCS_URL, params=params, timeout=120, stream=True
    )
    if resp.status_code != 200:
        raise RuntimeError(f"WCS returned HTTP {resp.status_code}")

    ct = resp.headers.get("Content-Type", "")
    if "tiff" not in ct.lower() and "octet" not in ct.lower():
        # If the WCS returns an XML exception, surface it
        preview = resp.text[:200] if hasattr(resp, "text") else ct
        raise RuntimeError(f"WCS non-TIFF response: {ct!r}. Body: {preview}")

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    _validate_raster(output_path)
    return True


def _try_inlp_arcgis(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Fetch INLP DTM via EPA ArcGIS REST ImageServer exportImage.

    Tries each endpoint in _INLP_ARCGIS_ENDPOINTS until one succeeds.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    bbox_str = f"{min_lon},{min_lat},{max_lon},{max_lat}"

    params = {
        "bbox":                  bbox_str,
        "bboxSR":                4326,
        "size":                  "1024,1024",
        "imageSR":               4326,
        "format":                "tiff",
        "pixelType":             "F32",
        "noDataInterpretation":  "esriNoDataMatchAny",
        "interpolation":         "+RSP_BilinearInterpolation",
        "f":                     "image",
    }

    for endpoint in _INLP_ARCGIS_ENDPOINTS:
        try:
            resp = requests.get(endpoint, params=params, timeout=60, stream=True)
            if resp.status_code != 200:
                log.debug(f"ArcGIS {endpoint}: HTTP {resp.status_code}")
                continue
            ct = resp.headers.get("Content-Type", "")
            if "tiff" not in ct.lower() and "image" not in ct.lower():
                log.debug(f"ArcGIS {endpoint}: non-image ({ct[:60]})")
                continue
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            if output_path.stat().st_size < 5000:
                output_path.unlink(missing_ok=True)
                continue
            _validate_raster(output_path)
            return True
        except Exception as e:
            log.debug(f"ArcGIS endpoint failed: {e}")
            output_path.unlink(missing_ok=True)

    raise RuntimeError("INLP ArcGIS: all endpoints failed")


def _try_mapzen_local(bbox_wgs84: list, output_path: Path) -> bool:
    """Delegate to Mapzen Terrarium download in lidar.py."""
    try:
        from pipeline.lidar import download_mapzen_dem
    except ImportError:
        from lidar import download_mapzen_dem
    return download_mapzen_dem(bbox_wgs84, output_path, zoom=12)


# ─── Informational: STAC catalogue query ─────────────────────────────────────

def query_inlp_stac(bbox_wgs84: list) -> list:
    """
    Query data.gov.ie CKAN API for INLP tile URLs covering the bbox.
    Returns list of download URLs (informational only — most require auth).
    """
    try:
        resp = requests.get(
            _INLP_STAC_API,
            params={
                "q":    "national lidar DTM",
                "rows": 20,
                "fq":   "organization:tailte-eireann",
            },
            timeout=30,
        )
        resp.raise_for_status()
        results = resp.json().get("result", {}).get("results", [])
        return [
            resource.get("url", "")
            for pkg in results
            for resource in pkg.get("resources", [])
            if resource.get("url", "").endswith((".tif", ".tiff", ".laz", ".las"))
        ]
    except Exception as e:
        log.warning(f"INLP STAC query failed: {e}")
        return []


# ─── Utilities ────────────────────────────────────────────────────────────────

def _validate_raster(path: Path) -> None:
    """Raise RuntimeError if raster is missing, empty, or all-nodata."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            arr    = src.read(1).astype(np.float32)
            nodata = src.nodata
        if arr.size == 0:
            raise ValueError("empty raster")
        if nodata is not None and np.all(arr == nodata):
            raise ValueError("all nodata — outside coverage")
    except Exception as e:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Raster validation failed: {e}") from e

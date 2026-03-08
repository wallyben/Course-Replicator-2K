"""
lidar.py — LiDAR / elevation data acquisition for Irish golf courses.

Priority chain:
  1. Tailte Éireann Irish National LiDAR Programme (INLP) — 0.5–1m, Ireland
  2. EU-DEM v1.1 via Copernicus / OpenTopography — 25m, full Europe
  3. SRTM 30m via OpenTopography API — 30m, global fallback

Output:
  - dtm.tif   — Digital Terrain Model clipped to buffered bounding box
  - coverage  — string describing which source was used
"""

import logging
import math
import os
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)


# ─── Public entry point ───────────────────────────────────────────────────────

def acquire_elevation(boundary_data: dict, output_dir: Path) -> Tuple[Path, str]:
    """
    Acquire best available elevation data for the given boundary.

    Args:
        boundary_data: Output from boundary.resolve_boundary()
        output_dir:    Directory to write dtm.tif into

    Returns:
        (dtm_path, coverage_description)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtm_path = output_dir / "dtm.tif"

    bbox = boundary_data["bbox_buffered_itm"]   # [minx, miny, maxx, maxy] ITM
    bbox_wgs84 = boundary_data["bbox_wgs84"]    # [minw, mins, maxe, maxn] WGS84

    # Try sources in order
    for source_fn, label in [
        (_try_inlp,       "Irish National LiDAR Programme (INLP) — 0.5m"),
        (_try_eudem,      "EU-DEM v1.1 (Copernicus) — 25m"),
        (_try_srtm,       "SRTM v3 (NASA) — 30m"),
        (_try_open_elev,  "Open-Elevation API — 90m"),
    ]:
        try:
            log.info(f"Trying LiDAR source: {label}")
            result = source_fn(bbox_wgs84, dtm_path)
            if result and dtm_path.exists():
                log.info(f"Elevation acquired from: {label}")
                return dtm_path, label
        except Exception as e:
            log.warning(f"Source failed ({label}): {e}")

    raise RuntimeError(
        "All elevation sources failed. Check your internet connection and "
        "verify the bounding box covers a valid location in Ireland."
    )


# ─── Source: INLP (Tailte Éireann) ───────────────────────────────────────────

def _try_inlp(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Attempt to fetch Irish National LiDAR Programme data.

    The INLP datasets are served via data.gov.ie and ArcGIS REST services.
    We query the open WCS endpoint for the DSM/DTM raster service.

    Returns True if successful, False/raises if not.
    """
    # Tailte Éireann WCS endpoint for lidar DTM
    # Service: National LiDAR Programme - Bare Earth (DTM) 0.5m
    WCS_BASE = "https://wms.tailte.ie/inspire/ows"

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    params = {
        "SERVICE":     "WCS",
        "VERSION":     "1.0.0",
        "REQUEST":     "GetCoverage",
        "COVERAGE":    "DTM",
        "CRS":         "EPSG:4326",
        "BBOX":        f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "WIDTH":       "2048",
        "HEIGHT":      "2048",
        "FORMAT":      "GeoTIFF",
    }

    resp = requests.get(WCS_BASE, params=params, timeout=120, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"INLP WCS returned HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if "tiff" not in content_type.lower() and "geotiff" not in content_type.lower():
        # Likely returned an error XML
        raise RuntimeError(f"INLP WCS returned non-TIFF content: {content_type}")

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    # Validate it's a real raster
    _validate_tif(output_path)
    return True


# ─── Source: EU-DEM ───────────────────────────────────────────────────────────

def _try_eudem(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Fetch EU-DEM 25m via OpenTopography hosted raster (publicly accessible GeoTIFF).
    Covers all of Ireland at 25m resolution.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, Resampling

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    # OpenTopography EU_DEM endpoint
    url = (
        "https://portal.opentopography.org/API/globaldem"
        "?demtype=SRTMGL1"  # Use SRTM as proxy — see _try_srtm for proper SRTM
    )
    # EU-DEM via direct Copernicus STAC is complex; use SRTM as the 30m fallback instead.
    # This function tries a direct STAC fetch of EU_DEM tiles.

    # EU-DEM tile naming: EU_DEM_be_{lat}_{lon} for 5° tiles
    tile_lat = int(math.floor(min_lat / 5)) * 5
    tile_lon = int(math.floor(min_lon / 5)) * 5
    ns = "N" if tile_lat >= 0 else "S"
    ew = "E" if tile_lon >= 0 else "W"
    tile_name = f"eu_dem_v11_{ns}{abs(tile_lat):02d}{ew}{abs(tile_lon):03d}.TIF"

    # Copernicus Land Monitoring Service
    base_url = "https://land.copernicus.eu/en/products/eu-dem/eu-dem-v1.1"
    tile_url = f"https://download.gisco.eu/collection/eu_dem/v1.1/eu_dem_v11_E30N20.TIF"

    # Direct download URL for Ireland tile (E30N20 covers British Isles/Ireland)
    ireland_tile_url = (
        "https://opentopography.s3.sdsc.edu/raster/EU_DEM/EU_DEM_be_5deg/"
        "eu_dem_v11_E30N20.TIF"
    )

    tmp_tile = output_path.parent / "eudem_tile.tif"
    _download_file(ireland_tile_url, tmp_tile, desc="EU-DEM tile")
    if not tmp_tile.exists():
        raise RuntimeError("EU-DEM tile download failed")

    # Clip to bounding box and write
    _clip_raster(tmp_tile, output_path, bbox_wgs84)
    tmp_tile.unlink(missing_ok=True)
    return True


# ─── Source: SRTM 30m ────────────────────────────────────────────────────────

def _try_srtm(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Fetch SRTM 30m via OpenTopography public API.
    Requires a free API key (config.SRTM_API_KEY) or works with empty key
    for low-volume usage.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    params = {
        "demtype":    "SRTMGL1",
        "south":      min_lat,
        "north":      max_lat,
        "west":       min_lon,
        "east":       max_lon,
        "outputFormat": "GTiff",
    }
    if config.SRTM_API_KEY:
        params["API_Key"] = config.SRTM_API_KEY

    resp = requests.get(
        config.SRTM_API_URL, params=params, timeout=120, stream=True
    )
    if resp.status_code != 200:
        raise RuntimeError(f"OpenTopography SRTM returned HTTP {resp.status_code}")

    content_type = resp.headers.get("Content-Type", "")
    if "tiff" not in content_type.lower():
        raise RuntimeError(f"OpenTopography returned non-TIFF: {content_type[:100]}")

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    _validate_tif(output_path)
    return True


# ─── Source: Open-Elevation API (90m SRTM, zero-dependency fallback) ─────────

def _try_open_elev(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Last-resort fallback: query Open-Elevation API for a grid of points,
    then write a synthetic GeoTIFF.

    Resolution: ~90m (sample at 0.001° intervals).
    This is coarse but produces a valid raster for the pipeline.
    """
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    step = 0.001  # ~90m
    lons = list(_frange(min_lon, max_lon, step))
    lats = list(_frange(min_lat, max_lat, step))

    if not lons or not lats:
        raise ValueError("Bounding box too small for open-elevation sampling")

    locations = [{"latitude": lat, "longitude": lon} for lat in lats for lon in lons]

    # Batch in chunks of 1000
    elevations = []
    batch_size = 1000
    for i in range(0, len(locations), batch_size):
        batch = locations[i: i + batch_size]
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": batch},
            timeout=60,
        )
        resp.raise_for_status()
        results = resp.json()["results"]
        elevations.extend(r["elevation"] for r in results)

    # Reconstruct 2D array
    ny, nx = len(lats), len(lons)
    arr = np.array(elevations, dtype=np.float32).reshape(ny, nx)
    arr = np.flipud(arr)  # rasterio origin is top-left

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, nx, ny)
    with rasterio.open(
        output_path, "w",
        driver="GTiff",
        height=ny, width=nx,
        count=1, dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(arr, 1)

    return True


# ─── Raster utilities ─────────────────────────────────────────────────────────

def _clip_raster(src_path: Path, dst_path: Path, bbox_wgs84: list) -> None:
    """Clip a GeoTIFF to a WGS84 bounding box."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box, mapping

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    clip_geom = [mapping(box(min_lon, min_lat, max_lon, max_lat))]

    with rasterio.open(src_path) as src:
        out_image, out_transform = rio_mask(src, clip_geom, crop=True)
        out_meta = src.meta.copy()

    out_meta.update({
        "driver":    "GTiff",
        "height":    out_image.shape[1],
        "width":     out_image.shape[2],
        "transform": out_transform,
    })

    with rasterio.open(dst_path, "w", **out_meta) as dst:
        dst.write(out_image)


def _validate_tif(path: Path) -> None:
    """Raise if path is not a readable raster with valid data."""
    import rasterio
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            if arr.size == 0:
                raise ValueError("Raster is empty")
            if np.all(arr == src.nodata):
                raise ValueError("Raster is all nodata — outside coverage")
    except Exception as e:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Raster validation failed: {e}") from e


def _download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
    """Stream-download a file with retry logic."""
    for attempt in range(4):
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(
                total=total, unit="B", unit_scale=True, desc=desc
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
            return
        except requests.RequestException as e:
            if attempt < 3:
                wait = 2 ** attempt
                log.warning(f"Download failed (attempt {attempt+1}): {e}. Retry in {wait}s")
                time.sleep(wait)
            else:
                raise


def _frange(start: float, stop: float, step: float):
    val = start
    while val < stop:
        yield round(val, 6)
        val += step

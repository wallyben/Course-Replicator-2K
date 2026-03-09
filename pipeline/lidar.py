"""
lidar.py — LiDAR / elevation data acquisition for Irish golf courses.

Priority chain:
  1. Tailte Éireann Irish National LiDAR Programme (INLP) — 0.5–1m, Ireland
  2. EU-DEM v1.1 via Copernicus / OpenTopography — 25m, full Europe
  3. SRTM 30m via OpenTopography API — 30m, global fallback
  4. Mapzen Terrarium tiles (AWS) — ~38m, global primary fallback  [V2 NEW]
  5. Open-Elevation API — 90m, zero-dependency last resort

Output:
  - dtm.tif   — Digital Terrain Model clipped to buffered bounding box
  - coverage  — string describing which source was used
"""

import io
import logging
import math
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

import sys
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
        (_try_mapzen,     "Mapzen Terrarium tiles (AWS) — 38m"),   # V2: reliable global fallback
        (_try_open_elev,  "Open-Elevation API — 90m"),
    ]:
        try:
            log.info(f"Trying elevation source: {label}")
            result = source_fn(bbox_wgs84, dtm_path)
            if result and dtm_path.exists():
                # V2: validate DEM integrity before accepting
                validate_dem_integrity(dtm_path)
                log.info(f"Elevation acquired from: {label}")
                return dtm_path, label
        except RuntimeError as e:
            log.warning(f"Source failed ({label}): {e}")
            if dtm_path.exists():
                dtm_path.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"Source failed ({label}): {e}")
            if dtm_path.exists():
                dtm_path.unlink(missing_ok=True)

    raise RuntimeError(
        "All elevation sources failed. Check your internet connection and "
        "verify the bounding box covers a valid location."
    )


# ─── V2: DEM integrity validation ─────────────────────────────────────────────

def validate_dem_integrity(dtm_path: Path) -> None:
    """
    FIX 2: Validate raw DEM has usable elevation data BEFORE clipping.

    This is called on the full downloaded DEM (not the clipped result).
    Threshold: ≥ 10% valid pixels required on the full tile.

    Raises:
        RuntimeError: if DEM is empty, too small, or entirely NaN/nodata.
    """
    import rasterio

    if not dtm_path.exists():
        raise RuntimeError("DEM contains no valid elevation data")

    with rasterio.open(dtm_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if arr.size == 0:
        raise RuntimeError("DEM contains no valid elevation data")

    if arr.shape[0] < 4 or arr.shape[1] < 4:
        raise RuntimeError("DEM contains no valid elevation data")

    if nodata is not None:
        arr = arr.copy()
        arr[arr == nodata] = np.nan

    if np.all(np.isnan(arr)):
        raise RuntimeError("DEM contains no valid elevation data")

    valid_count = int(np.sum(~np.isnan(arr)))
    total = arr.size
    valid_pct = valid_count / total * 100
    if valid_pct < 10.0:
        raise RuntimeError(
            f"DEM contains no valid elevation data "
            f"(only {valid_pct:.1f}% valid pixels)"
        )

    log.debug(f"DEM integrity OK: {arr.shape}, {valid_pct:.1f}% valid")


def validate_dem_post_clip(dtm_path: Path) -> None:
    """
    Lenient DEM validation for post-clip rasters.

    Problem 2: After clipping a Mapzen mosaic to a course bbox, valid
    coverage can legitimately be lower (e.g. coastal courses, small area).
    This check accepts as little as 5% valid pixels but still rejects
    completely empty rasters.

    Raises:
        RuntimeError: only if the clipped raster has no valid data at all.
    """
    import rasterio

    if not dtm_path.exists():
        raise RuntimeError("DEM contains no valid elevation data")

    with rasterio.open(dtm_path) as src:
        arr = src.read(1).astype(np.float32)
        nodata = src.nodata

    if arr.size == 0:
        raise RuntimeError("DEM contains no valid elevation data")

    if arr.shape[0] < 2 or arr.shape[1] < 2:
        raise RuntimeError("DEM contains no valid elevation data")

    if nodata is not None:
        arr = arr.copy()
        arr[arr == nodata] = np.nan

    if np.all(np.isnan(arr)):
        raise RuntimeError("DEM contains no valid elevation data")

    valid_pct = float(np.sum(~np.isnan(arr))) / arr.size * 100

    if valid_pct < 5.0:
        # Attempt median fill on very sparse data rather than hard-failing.
        # If we have SOME valid pixels, terrain.py can fill the rest.
        log.warning(
            f"Post-clip DEM is sparse ({valid_pct:.1f}% valid pixels). "
            "Terrain NaN fill will interpolate missing cells."
        )
    else:
        log.debug(f"Post-clip DEM coverage OK: {valid_pct:.1f}% valid")


# ─── Source: INLP (Tailte Éireann) ───────────────────────────────────────────

def _try_inlp(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Attempt to fetch Irish National LiDAR Programme data.

    The INLP datasets are served via data.gov.ie and ArcGIS REST services.
    We query the open WCS endpoint for the DSM/DTM raster service.

    Returns True if successful, False/raises if not.
    """
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
        raise RuntimeError(f"INLP WCS returned non-TIFF content: {content_type}")

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    _validate_tif(output_path)
    return True


# ─── Source: EU-DEM (FIX 4 — corrected tile resolution) ──────────────────────

def _try_eudem(bbox_wgs84: list, output_path: Path) -> bool:
    """
    FIX 4: Fetch EU-DEM 25m using correct ETRS89-LAEA 1000km tile grid.

    Tile naming: eu_dem_v11_E{XX}N{YY}.TIF
    Where XX = floor(LAEA_easting / 1_000_000) * 10
          YY = floor(LAEA_northing / 1_000_000) * 10

    Retries up to 3 times per candidate tile.
    """
    from pyproj import Transformer

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # Convert centroid to ETRS89-LAEA (EPSG:3035)
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    x_laea, y_laea = t.transform(center_lon, center_lat)

    # Tile grid: 1000km × 1000km, named by dividing by 100_000 (gives 2-digit 100km unit)
    tile_e = int(x_laea // 100000)
    tile_n = int(y_laea // 100000)
    primary_tile = f"eu_dem_v11_E{tile_e}N{tile_n}.TIF"

    # Generate candidate tiles — try neighbours if primary fails
    candidate_tiles = [
        primary_tile,
        f"eu_dem_v11_E{tile_e-1}N{tile_n}.TIF",
        f"eu_dem_v11_E{tile_e}N{tile_n-1}.TIF",
        f"eu_dem_v11_E{tile_e+1}N{tile_n}.TIF",
        # Known Ireland tile as final fallback
        "eu_dem_v11_E30N20.TIF",
    ]
    # Deduplicate preserving order
    seen = set()
    candidates = []
    for t_name in candidate_tiles:
        if t_name not in seen:
            seen.add(t_name)
            candidates.append(t_name)

    base_url = f"{config.EUDEM_BASE_URL}"
    tmp_tile = output_path.parent / "eudem_tile.tif"

    for tile_name in candidates:
        tile_url = f"{base_url}/{tile_name}"
        log.debug(f"EU-DEM trying tile: {tile_name}")

        for attempt in range(3):
            try:
                _download_file(tile_url, tmp_tile, desc=f"EU-DEM {tile_name}")
                if tmp_tile.exists() and tmp_tile.stat().st_size > 10000:
                    _clip_raster(tmp_tile, output_path, bbox_wgs84)
                    tmp_tile.unlink(missing_ok=True)
                    return True
                else:
                    tmp_tile.unlink(missing_ok=True)
                    break
            except Exception as e:
                tmp_tile.unlink(missing_ok=True)
                if attempt < 2:
                    wait = 2 ** attempt
                    log.debug(f"EU-DEM attempt {attempt+1} failed: {e}. Retry in {wait}s")
                    time.sleep(wait)
                else:
                    log.debug(f"EU-DEM tile {tile_name} exhausted retries: {e}")

    raise RuntimeError(f"EU-DEM: all candidate tiles failed for centroid ({center_lon:.3f},{center_lat:.3f})")


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


# ─── Source: Mapzen Terrarium tiles (FIX 1 — reliable global fallback) ───────

def _try_mapzen(bbox_wgs84: list, output_path: Path) -> bool:
    """
    FIX 1: Download Mapzen Terrarium elevation tiles from AWS S3.

    URL: https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
    Encoding: elevation = (R * 256 + G + B / 256) - 32768 metres
    Zoom 12 ≈ 38m/pixel resolution — sufficient for terrain modelling.
    """
    return download_mapzen_dem(bbox_wgs84, output_path)


def download_mapzen_dem(bbox_wgs84: list, output_path: Path, zoom: int = 12) -> bool:
    """
    Download Mapzen Terrarium tiles, decode elevation, merge into GeoTIFF.

    Args:
        bbox_wgs84:  [min_lon, min_lat, max_lon, max_lat]
        output_path: Destination GeoTIFF path
        zoom:        Tile zoom level (12 ≈ 38m/px, 13 ≈ 19m/px)

    Returns:
        True on success, raises RuntimeError on failure.
    """
    try:
        import mercantile
    except ImportError:
        raise RuntimeError("mercantile not installed — pip install mercantile")

    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    from rasterio.merge import merge as rio_merge
    from PIL import Image

    TERRARIUM_URL = (
        "https://s3.amazonaws.com/elevation-tiles-prod/terrarium"
        "/{z}/{x}/{y}.png"
    )

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))

    if not tiles:
        raise RuntimeError("No Mapzen tiles found for bounding box")

    # Problem 2 guard: if too many tiles, step down zoom to keep it manageable
    if len(tiles) > 100 and zoom > 9:
        log.warning(
            f"Mapzen: {len(tiles)} tiles at zoom {zoom} — stepping down to zoom {zoom-1}"
        )
        return download_mapzen_dem(bbox_wgs84, output_path, zoom=zoom - 1)

    log.info(f"Mapzen: downloading {len(tiles)} tiles at zoom {zoom}")

    tmp_dir = output_path.parent / "_mapzen_tiles"
    tmp_dir.mkdir(exist_ok=True)
    tile_paths = []

    session = requests.Session()
    session.headers.update({"User-Agent": "CourseReplicator2K/2.0"})

    for tile in tiles:
        url = TERRARIUM_URL.format(z=tile.z, x=tile.x, y=tile.y)
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                log.debug(f"Mapzen tile {tile} returned HTTP {resp.status_code}")
                continue

            # Decode Terrarium PNG → elevation metres
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            elevation = arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0 - 32768.0

            # Sea-level clamp: water bodies sometimes encode slightly below -32768
            elevation = np.clip(elevation, -500.0, 9000.0)

            bounds = mercantile.bounds(tile)
            tile_transform = from_bounds(
                bounds.west, bounds.south, bounds.east, bounds.north,
                elevation.shape[1], elevation.shape[0]
            )

            tile_path = tmp_dir / f"tile_{tile.z}_{tile.x}_{tile.y}.tif"
            with rasterio.open(
                tile_path, "w",
                driver="GTiff",
                height=elevation.shape[0],
                width=elevation.shape[1],
                count=1,
                dtype="float32",
                crs=CRS.from_epsg(4326),
                transform=tile_transform,
                nodata=-32768.0,
            ) as dst:
                dst.write(elevation[np.newaxis, :, :])

            tile_paths.append(tile_path)

        except Exception as e:
            log.debug(f"Mapzen tile {tile} failed: {e}")
            continue

    if not tile_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Mapzen: no tiles downloaded successfully")

    # Merge all tiles into single raster
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        merged_arr, merged_transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height":    merged_arr.shape[1],
            "width":     merged_arr.shape[2],
            "transform": merged_transform,
            "driver":    "GTiff",
        })
    finally:
        for src in srcs:
            src.close()

    merged_path = tmp_dir / "merged.tif"
    with rasterio.open(merged_path, "w", **meta) as dst:
        dst.write(merged_arr)

    # Clip merged raster to requested bbox
    _clip_raster(merged_path, output_path, bbox_wgs84)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    log.info(f"Mapzen DEM written: {output_path}")
    return True


# ─── Source: Open-Elevation API (90m SRTM, zero-dependency last resort) ──────

def _try_open_elev(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Last-resort fallback: query Open-Elevation API for a grid of points,
    then write a synthetic GeoTIFF.

    Resolution: ~90m (sample at 0.001° intervals).
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
            if src.nodata is not None and np.all(arr == src.nodata):
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

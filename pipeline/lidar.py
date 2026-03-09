"""
lidar.py — Multi-source elevation data acquisition.

Priority chain (controlled by config.DEM_SOURCES):
  1. INLP — Irish National LiDAR Programme (Tailte Éireann) — 0.5m, Ireland only
  2. OpenTopography — LiDAR/DEM global API, 1–30m
  3. Copernicus GLO-30 — 30m global (AWS Open Data)
  4. SRTM v3 (NASA) via OpenTopography — 30m global
  5. Mapzen Terrarium tiles (AWS) — ~38m global final fallback

Step 8: After download, every DEM is validated for:
  - resolution / pixel count
  - variance (reject flat/corrupt)
  - slope realism (reject spike artifacts)

Output:
  dtm.tif — clipped to buffered course bbox
  coverage — string describing which source was used
"""

import io
import logging
import math
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

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


# ─── Public entry point ───────────────────────────────────────────────────────

def acquire_elevation(boundary_data: dict, output_dir: Path) -> Tuple[Path, str]:
    """
    Acquire best available elevation data for the course boundary.

    Sources are tried in the order specified by config.DEM_SOURCES.
    Each source must produce a valid GeoTIFF that passes quality validation.

    Args:
        boundary_data: Output from boundary.resolve_boundary()
        output_dir:    Directory to write dtm.tif into

    Returns:
        (dtm_path, coverage_description)

    Raises:
        RuntimeError: if all sources fail
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtm_path   = output_dir / "dtm.tif"

    bbox_wgs84 = boundary_data["bbox_wgs84"]   # [min_lon, min_lat, max_lon, max_lat]

    # Build ordered source list from config
    _HANDLERS = {
        "inlp":           (_try_inlp,           "Irish National LiDAR (INLP) — 0.5m"),
        "opentopography": (_try_opentopography,  "OpenTopography global DEM — 1-30m"),
        "copernicus":     (_try_copernicus_glo30,"Copernicus GLO-30 — 30m"),
        "srtm":           (_try_srtm,            "NASA SRTM v3 — 30m"),
        "mapzen":         (_try_mapzen,          "Mapzen Terrarium — ~38m"),
    }

    sources = getattr(config, "DEM_SOURCES",
                      ["inlp", "opentopography", "copernicus", "srtm", "mapzen"])

    for key in sources:
        if key not in _HANDLERS:
            log.warning(f"Unknown DEM source '{key}' in config.DEM_SOURCES — skipped")
            continue
        fn, label = _HANDLERS[key]
        log.info(f"  Trying DEM source: {label}")
        try:
            ok = fn(bbox_wgs84, dtm_path)
            if ok and dtm_path.exists():
                validate_dem_integrity(dtm_path)
                stats = _dem_quality_stats(dtm_path)
                log.info(
                    f"  DEM acquired: {label}  "
                    f"({stats['rows']}×{stats['cols']} px, "
                    f"variance={stats['variance_m2']:.2f} m², "
                    f"max_slope={stats['max_slope_deg']:.1f}°)"
                )
                return dtm_path, label
        except RuntimeError as e:
            log.warning(f"  Source failed ({label}): {e}")
            dtm_path.unlink(missing_ok=True)
        except Exception as e:
            log.warning(f"  Source error ({label}): {e}")
            dtm_path.unlink(missing_ok=True)

    raise RuntimeError(
        "All elevation sources exhausted. Check network connectivity and "
        "verify the bounding box covers a valid location."
    )


# ─── Step 8: DEM quality validation ──────────────────────────────────────────

def validate_dem_integrity(dtm_path: Path) -> None:
    """
    Validate DEM has usable data BEFORE accepting it.

    Checks:
    1. File exists and opens as a raster
    2. Grid is at least 4×4 pixels
    3. ≥ DEM_MIN_VALID_PCT % of pixels are non-nodata
    4. Elevation variance > DEM_MIN_VARIANCE (rejects uniform/flat corrupt DEMs)

    Raises RuntimeError on any failure.
    """
    import rasterio

    if not dtm_path.exists():
        raise RuntimeError("DEM file does not exist")

    try:
        with rasterio.open(dtm_path) as src:
            arr    = src.read(1).astype(np.float32)
            nodata = src.nodata
    except Exception as e:
        raise RuntimeError(f"Cannot open DEM: {e}") from e

    if arr.size == 0 or arr.shape[0] < 4 or arr.shape[1] < 4:
        raise RuntimeError(f"DEM too small: {arr.shape}")

    valid = arr.copy()
    if nodata is not None:
        valid[arr == nodata] = np.nan

    n_valid = int(np.sum(~np.isnan(valid)))
    pct     = n_valid / arr.size * 100
    min_pct = getattr(config, "DEM_MIN_VALID_PCT", 10.0)

    if pct < min_pct:
        raise RuntimeError(
            f"DEM only {pct:.1f}% valid pixels (need ≥ {min_pct:.0f}%)"
        )

    finite = valid[~np.isnan(valid)]
    variance = float(np.var(finite)) if len(finite) > 1 else 0.0
    min_var  = getattr(config, "DEM_MIN_VARIANCE", 0.1)
    if variance < min_var:
        raise RuntimeError(
            f"DEM appears flat/corrupt: variance={variance:.4f} m² < {min_var}"
        )

    log.debug(f"DEM integrity OK: {arr.shape}, {pct:.1f}% valid, var={variance:.2f}")


def validate_dem_post_clip(dtm_path: Path) -> None:
    """
    Lenient validation for post-clip DEMs (coastal courses may be sparse).
    Warns rather than raises when coverage is low.
    """
    import rasterio
    if not dtm_path.exists():
        raise RuntimeError("Clipped DEM does not exist")

    with rasterio.open(dtm_path) as src:
        arr    = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    pct = float(np.sum(~np.isnan(arr))) / arr.size * 100 if arr.size else 0
    min_pct = getattr(config, "DEM_POST_CLIP_PCT", 5.0)
    if pct < min_pct:
        log.warning(
            f"Post-clip DEM sparse ({pct:.1f}% valid). "
            "terrain.py NaN-fill will interpolate gaps."
        )
    else:
        log.debug(f"Post-clip DEM OK: {pct:.1f}% valid")


def _dem_quality_stats(dtm_path: Path) -> dict:
    """Return dict with rows, cols, variance_m2, max_slope_deg for logging."""
    import rasterio
    try:
        with rasterio.open(dtm_path) as src:
            arr    = src.read(1).astype(np.float32)
            nodata = src.nodata
            res_x  = abs(src.transform.a)
        if nodata is not None:
            arr[arr == nodata] = np.nan
        finite = arr[~np.isnan(arr)]
        variance = float(np.var(finite)) if len(finite) > 1 else 0.0
        # Approximate max slope from central differences
        gy, gx = np.gradient(np.nan_to_num(arr))
        # res_x in degrees → convert to metres (1° ≈ 111 km)
        m_per_px = res_x * 111_000
        slope_rad = np.arctan(np.sqrt(gx**2 + gy**2) / m_per_px)
        max_slope = float(np.nanmax(np.degrees(slope_rad)))
        return {
            "rows": arr.shape[0], "cols": arr.shape[1],
            "variance_m2": variance, "max_slope_deg": max_slope,
        }
    except Exception:
        return {"rows": 0, "cols": 0, "variance_m2": 0.0, "max_slope_deg": 0.0}


# ─── Source 1: INLP (Tailte Éireann) ─────────────────────────────────────────

def _try_inlp(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Try Irish National LiDAR Programme via WCS then ArcGIS ImageServer.
    Returns True on success, raises RuntimeError if outside Ireland or unavailable.
    """
    # Delegate to the dedicated Ireland module
    try:
        from pipeline.lidar_ireland import acquire_irish_lidar, is_ireland
    except ImportError:
        from lidar_ireland import acquire_irish_lidar, is_ireland

    if not is_ireland(bbox_wgs84):
        raise RuntimeError("Course is outside Ireland — INLP not applicable")

    result_path, label = acquire_irish_lidar(bbox_wgs84, output_path.parent)
    if result_path and result_path.exists():
        # lidar_ireland writes directly to dtm.tif — just confirm
        if result_path != output_path:
            import shutil
            shutil.copy2(result_path, output_path)
        return True
    raise RuntimeError(f"INLP acquisition failed: {label}")


# ─── Source 2: OpenTopography global DEM ────────────────────────────────────

def _try_opentopography(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Download DEM from OpenTopography public API.

    Tries DEM types in order: SRTMGL1 (30m), SRTMGL3 (90m), COP30 (30m).
    A free API key (config.OPENTOPO_API_KEY) is optional but gives higher limits.
    """
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    api_url = getattr(config, "OPENTOPO_API_URL",
                      "https://portal.opentopography.org/API/globaldem")
    api_key = getattr(config, "OPENTOPO_API_KEY", "") or \
              getattr(config, "SRTM_API_KEY", "")

    dem_types = ["SRTMGL1", "AW3D30", "SRTMGL3"]

    for dem_type in dem_types:
        params = {
            "demtype":      dem_type,
            "south":        min_lat,
            "north":        max_lat,
            "west":         min_lon,
            "east":         max_lon,
            "outputFormat": "GTiff",
        }
        if api_key:
            params["API_Key"] = api_key

        try:
            resp = requests.get(api_url, params=params, timeout=120, stream=True)
            if resp.status_code == 401:
                log.debug(f"OpenTopography {dem_type}: auth required, skipping")
                continue
            if resp.status_code != 200:
                log.debug(f"OpenTopography {dem_type}: HTTP {resp.status_code}")
                continue
            ct = resp.headers.get("Content-Type", "")
            if "tiff" not in ct.lower() and "octet" not in ct.lower():
                log.debug(f"OpenTopography {dem_type}: non-TIFF response ({ct[:60]})")
                continue
            with open(output_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            if output_path.stat().st_size < 1000:
                output_path.unlink(missing_ok=True)
                continue
            _validate_tif(output_path)
            log.info(f"  OpenTopography source: {dem_type}")
            return True
        except Exception as e:
            log.debug(f"OpenTopography {dem_type} failed: {e}")
            output_path.unlink(missing_ok=True)

    raise RuntimeError("OpenTopography: all DEM types failed")


# ─── Source 3: Copernicus GLO-30 ─────────────────────────────────────────────

def _try_copernicus_glo30(bbox_wgs84: list, output_path: Path) -> bool:
    """
    Download Copernicus GLO-30 DEM tiles from the AWS Open Data Registry.

    Tile naming: Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM.tif
    Covers 1°×1° at 30m resolution; stored as Cloud-Optimised GeoTIFFs on S3.
    """
    import rasterio
    from rasterio.merge import merge as rio_merge
    from rasterio.crs import CRS
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box, mapping

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    base_url = getattr(config, "COPERNICUS_DEM_URL",
                       "https://copernicus-dem-30m.s3.amazonaws.com")

    # Build list of 1°×1° tiles that intersect the bbox
    tile_lons = range(int(math.floor(min_lon)), int(math.ceil(max_lon)))
    tile_lats = range(int(math.floor(min_lat)), int(math.ceil(max_lat)))

    tmp_dir    = output_path.parent / "_cop30_tiles"
    tmp_dir.mkdir(exist_ok=True)
    tile_paths = []
    session    = requests.Session()
    session.headers.update({"User-Agent": "CourseReplicator2K/2.0"})

    for tlat in tile_lats:
        for tlon in tile_lons:
            lat_tag  = f"N{abs(tlat):02d}" if tlat >= 0 else f"S{abs(tlat):02d}"
            lon_tag  = f"E{abs(tlon):03d}" if tlon >= 0 else f"W{abs(tlon):03d}"
            filename = (
                f"Copernicus_DSM_COG_10_{lat_tag}_00_{lon_tag}_00_DEM/"
                f"Copernicus_DSM_COG_10_{lat_tag}_00_{lon_tag}_00_DEM.tif"
            )
            url = f"{base_url}/{filename}"

            tile_path = tmp_dir / f"cop30_{lat_tag}_{lon_tag}.tif"
            for attempt in range(3):
                try:
                    resp = session.get(url, timeout=60, stream=True)
                    if resp.status_code == 404:
                        log.debug(f"Copernicus GLO-30: tile not found ({lat_tag},{lon_tag})")
                        break
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}")
                    with open(tile_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=65536):
                            f.write(chunk)
                    if tile_path.stat().st_size > 1000:
                        tile_paths.append(tile_path)
                        break
                    else:
                        tile_path.unlink(missing_ok=True)
                        break
                except Exception as e:
                    tile_path.unlink(missing_ok=True)
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        log.debug(f"Copernicus GLO-30 tile {lat_tag},{lon_tag} failed: {e}")

    if not tile_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Copernicus GLO-30: no tiles downloaded")

    # Merge tiles
    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        merged_arr, merged_transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height":    merged_arr.shape[1],
            "width":     merged_arr.shape[2],
            "transform": merged_transform,
            "driver":    "GTiff",
            "compress":  "lzw",
        })
    finally:
        for src in srcs:
            src.close()

    merged_path = tmp_dir / "merged.tif"
    with rasterio.open(merged_path, "w", **meta) as dst:
        dst.write(merged_arr)

    # Clip to bbox
    _clip_raster(merged_path, output_path, bbox_wgs84)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.info("  Copernicus GLO-30: merged and clipped")
    return True


# ─── Source 4: NASA SRTM 30m ─────────────────────────────────────────────────

def _try_srtm(bbox_wgs84: list, output_path: Path) -> bool:
    """Download SRTM v3 30m via OpenTopography API."""
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    api_url = getattr(config, "SRTM_API_URL",
                      "https://portal.opentopography.org/API/globaldem")
    api_key = getattr(config, "SRTM_API_KEY", "") or \
              getattr(config, "OPENTOPO_API_KEY", "")

    params = {
        "demtype":      "SRTMGL1",
        "south":        min_lat,
        "north":        max_lat,
        "west":         min_lon,
        "east":         max_lon,
        "outputFormat": "GTiff",
    }
    if api_key:
        params["API_Key"] = api_key

    resp = requests.get(api_url, params=params, timeout=120, stream=True)
    if resp.status_code != 200:
        raise RuntimeError(f"SRTM API returned HTTP {resp.status_code}")

    ct = resp.headers.get("Content-Type", "")
    if "tiff" not in ct.lower() and "octet" not in ct.lower():
        raise RuntimeError(f"SRTM returned non-TIFF ({ct[:80]})")

    with open(output_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)

    _validate_tif(output_path)
    return True


# ─── Source 5: Mapzen Terrarium tiles ────────────────────────────────────────

def _try_mapzen(bbox_wgs84: list, output_path: Path) -> bool:
    return download_mapzen_dem(bbox_wgs84, output_path)


def download_mapzen_dem(bbox_wgs84: list, output_path: Path, zoom: int = 12) -> bool:
    """
    Download Mapzen Terrarium PNG tiles, decode elevation, write GeoTIFF.

    Encoding: elevation = R*256 + G + B/256 - 32768 metres
    Zoom 12 ≈ 38m/pixel; zoom 13 ≈ 19m/pixel.
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

    tile_url = getattr(config, "MAPZEN_TILE_URL",
                       "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png")

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))
    if not tiles:
        raise RuntimeError("No Mapzen tiles for bounding box")

    if len(tiles) > 100 and zoom > 9:
        log.warning(f"Mapzen: {len(tiles)} tiles at z{zoom} — stepping down")
        return download_mapzen_dem(bbox_wgs84, output_path, zoom=zoom - 1)

    log.info(f"  Mapzen: {len(tiles)} tiles at zoom {zoom}")
    tmp_dir = output_path.parent / "_mapzen_tiles"
    tmp_dir.mkdir(exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "CourseReplicator2K/2.0"})

    tile_paths = []
    for tile in tiles:
        url = tile_url.format(z=tile.z, x=tile.x, y=tile.y)
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            elev = arr[:, :, 0] * 256.0 + arr[:, :, 1] + arr[:, :, 2] / 256.0 - 32768.0
            elev = np.clip(elev, -500.0, 9000.0)
            bounds    = mercantile.bounds(tile)
            transform = from_bounds(
                bounds.west, bounds.south, bounds.east, bounds.north,
                elev.shape[1], elev.shape[0]
            )
            tp = tmp_dir / f"t_{tile.z}_{tile.x}_{tile.y}.tif"
            with rasterio.open(
                tp, "w", driver="GTiff",
                height=elev.shape[0], width=elev.shape[1],
                count=1, dtype="float32",
                crs=CRS.from_epsg(4326),
                transform=transform, nodata=-32768.0,
            ) as dst:
                dst.write(elev[np.newaxis, :, :])
            tile_paths.append(tp)
        except Exception as e:
            log.debug(f"Mapzen tile {tile} failed: {e}")

    if not tile_paths:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError("Mapzen: no tiles downloaded")

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        merged_arr, merged_transform = rio_merge(srcs)
        meta = srcs[0].meta.copy()
        meta.update({
            "height":    merged_arr.shape[1],
            "width":     merged_arr.shape[2],
            "transform": merged_transform,
        })
    finally:
        for src in srcs:
            src.close()

    merged_path = tmp_dir / "merged.tif"
    with rasterio.open(merged_path, "w", **meta) as dst:
        dst.write(merged_arr)

    _clip_raster(merged_path, output_path, bbox_wgs84)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    log.info(f"  Mapzen DEM written: {output_path.name}")
    return True


# ─── Last-resort: Open-Elevation API (90m) ───────────────────────────────────

def _try_open_elev(bbox_wgs84: list, output_path: Path) -> bool:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    step = 0.001   # ~90m
    lons = list(_frange(min_lon, max_lon, step))
    lats = list(_frange(min_lat, max_lat, step))
    if not lons or not lats:
        raise ValueError("Bbox too small for open-elevation sampling")

    locations = [{"latitude": lat, "longitude": lon}
                 for lat in lats for lon in lons]

    elevations = []
    for i in range(0, len(locations), 1000):
        resp = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations[i:i+1000]},
            timeout=60,
        )
        resp.raise_for_status()
        elevations.extend(r["elevation"] for r in resp.json()["results"])

    ny, nx = len(lats), len(lons)
    arr = np.flipud(np.array(elevations, dtype=np.float32).reshape(ny, nx))
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, nx, ny)
    with rasterio.open(
        output_path, "w", driver="GTiff",
        height=ny, width=nx, count=1, dtype="float32",
        crs=CRS.from_epsg(4326), transform=transform, nodata=-9999,
    ) as dst:
        dst.write(arr, 1)
    return True


# ─── Raster utilities ─────────────────────────────────────────────────────────

def _clip_raster(src_path: Path, dst_path: Path, bbox_wgs84: list) -> None:
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import box, mapping

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    geom = [mapping(box(min_lon, min_lat, max_lon, max_lat))]

    with rasterio.open(src_path) as src:
        out_image, out_transform = rio_mask(src, geom, crop=True)
        meta = src.meta.copy()

    meta.update({
        "driver":    "GTiff",
        "height":    out_image.shape[1],
        "width":     out_image.shape[2],
        "transform": out_transform,
        "compress":  "lzw",
    })
    with rasterio.open(dst_path, "w", **meta) as dst:
        dst.write(out_image)


def _validate_tif(path: Path) -> None:
    import rasterio
    try:
        with rasterio.open(path) as src:
            arr = src.read(1)
            if arr.size == 0:
                raise ValueError("empty raster")
            if src.nodata is not None and np.all(arr == src.nodata):
                raise ValueError("all nodata — outside coverage area")
    except Exception as e:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Raster validation failed: {e}") from e


def _download_file(url: str, dest: Path, desc: str = "Downloading") -> None:
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
                time.sleep(2 ** attempt)
                log.warning(f"Download retry {attempt+1}: {e}")
            else:
                raise


def _try_eudem(bbox_wgs84: list, output_path: Path) -> bool:
    """EU-DEM v1.1 fallback (kept for lidar_ireland.py backward compat)."""
    from pyproj import Transformer

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    cx = (min_lon + max_lon) / 2
    cy = (min_lat + max_lat) / 2
    t  = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    xe, yn = t.transform(cx, cy)

    tile_e = int(xe // 100000)
    tile_n = int(yn // 100000)
    primary = f"eu_dem_v11_E{tile_e}N{tile_n}.TIF"
    candidates = [
        primary,
        f"eu_dem_v11_E{tile_e-1}N{tile_n}.TIF",
        f"eu_dem_v11_E{tile_e}N{tile_n-1}.TIF",
        "eu_dem_v11_E30N20.TIF",
    ]

    base_url = getattr(config, "EUDEM_BASE_URL",
                       "https://opentopography.s3.sdsc.edu/raster/EU_DEM/EU_DEM_be_5deg")
    tmp = output_path.parent / "_eudem_tile.tif"

    for name in dict.fromkeys(candidates):
        url = f"{base_url}/{name}"
        for attempt in range(3):
            try:
                _download_file(url, tmp, desc=f"EU-DEM {name}")
                if tmp.exists() and tmp.stat().st_size > 10_000:
                    _clip_raster(tmp, output_path, bbox_wgs84)
                    tmp.unlink(missing_ok=True)
                    return True
                tmp.unlink(missing_ok=True)
                break
            except Exception as e:
                tmp.unlink(missing_ok=True)
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    log.debug(f"EU-DEM {name}: {e}")

    raise RuntimeError("EU-DEM: all tiles failed")


def _frange(start: float, stop: float, step: float):
    val = start
    while val < stop:
        yield round(val, 6)
        val += step

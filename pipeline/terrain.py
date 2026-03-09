"""
terrain.py — DTM processing: heightmap generation, slope analysis, terrain classification.

Inputs:  dtm.tif (from lidar.py)
Outputs:
  - heightmap.png        16-bit greyscale PNG, normalised 0–65535
  - slope_map.png        Colour-shaded slope classification map
  - terrain_regions.tif  Raster: 1=flat, 2=gentle, 3=moderate, 4=steep
  - terrain_regions.geojson  Vector polygons per terrain class (V2)
  - terrain_stats.json   {z_min, z_max, z_range, slope_mean, resolution_m, ...}

V2 hardening:
  - validate_dem_integrity called before processing
  - NaN cells filled with spatial median before gradient (not mean)
  - Gaussian smoothing applied to reduce sensor noise
  - terrain_regions.geojson exported as vector polygons
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask as rio_mask
from rasterio.warp import calculate_default_transform, reproject
from rasterio.crs import CRS
from shapely.geometry import shape, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)


# ─── Main entry point ─────────────────────────────────────────────────────────

def process_terrain(dtm_path: Path, boundary_data: dict, output_dir: Path) -> dict:
    """
    Full terrain processing pipeline.

    Args:
        dtm_path:      Path to raw DTM GeoTIFF (any CRS)
        boundary_data: Output from boundary.resolve_boundary()
        output_dir:    Directory for output files

    Returns:
        terrain_stats dict with keys used downstream

    V2 hardening order:
        1. Validate raw DEM integrity BEFORE any reprojection or clipping
        2. Reproject to ITM
        3. Validate reprojected DEM
        4. Clip with progressive buffer fallback
        5. Validate clipped DEM (lenient — sparse is OK after clipping)
        6. Fill NaN → Gaussian smooth → slope → classify
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from pipeline.lidar import validate_dem_integrity, validate_dem_post_clip

    # ── STAGE A: Validate raw DEM BEFORE any processing ───────────────────────
    log.info("Validating raw DEM integrity...")
    validate_dem_integrity(dtm_path)   # raises RuntimeError if empty/all-NaN/<4x4

    # ── STAGE B: Reproject DTM to ITM ─────────────────────────────────────────
    dtm_itm_path = output_dir / "dtm_itm.tif"
    log.info("Reprojecting DTM to ITM...")
    _reproject_to_itm(dtm_path, dtm_itm_path)

    # ── STAGE C: Validate reprojected DEM (before clipping) ───────────────────
    log.info("Validating reprojected DEM...")
    validate_dem_integrity(dtm_itm_path)

    # ── STAGE D: Clip with progressive buffer reduction ────────────────────────
    dtm_clipped_path = output_dir / "dtm_clipped.tif"
    boundary_poly = shape(boundary_data["boundary_wgs84"])
    log.info("Clipping DTM to course boundary (with progressive buffer fallback)...")
    _clip_to_boundary_with_retry(dtm_itm_path, dtm_clipped_path, boundary_poly, boundary_data)

    # ── STAGE E: Validate clipped DEM (lenient — allows sparse data) ──────────
    log.info("Validating clipped DEM coverage...")
    validate_dem_post_clip(dtm_clipped_path)

    # ── STAGE F: Read and process ─────────────────────────────────────────────
    with rasterio.open(dtm_clipped_path) as src:
        dtm_arr = src.read(1).astype(np.float32)
        nodata  = src.nodata
        res_m   = src.res[0]
        transform = src.transform
        crs       = src.crs

    # Mask nodata
    if nodata is not None:
        dtm_arr[dtm_arr == nodata] = np.nan

    # Fill NaN → smooth → slope
    dtm_arr = _median_fill_nan(dtm_arr)
    dtm_arr = _gaussian_smooth(dtm_arr)

    # Step 4: Compute statistics
    valid = dtm_arr[~np.isnan(dtm_arr)]
    z_min  = float(np.nanmin(dtm_arr))
    z_max  = float(np.nanmax(dtm_arr))
    z_range = z_max - z_min
    z_mean = float(np.nanmean(dtm_arr))

    log.info(f"Elevation range: {z_min:.1f}m – {z_max:.1f}m ({z_range:.1f}m total relief)")

    if z_range < config.MIN_ELEVATION_RELIEF_M:
        log.warning(
            f"Very low terrain relief ({z_range:.2f}m). "
            "Course may be nearly flat or DTM quality is poor."
        )

    # Step 5: Generate heightmap PNG
    heightmap_path = output_dir / "heightmap.png"
    log.info("Generating heightmap PNG...")
    _generate_heightmap(dtm_arr, heightmap_path)

    # Step 6: Compute slope
    log.info("Computing slope map...")
    slope_arr = _compute_slope(dtm_arr, res_m)

    # Step 7: Terrain region classification
    regions_arr = _classify_terrain(slope_arr)
    regions_path = output_dir / "terrain_regions.tif"
    _write_raster(regions_arr.astype(np.uint8), regions_path, dtm_clipped_path)

    # V2: Export terrain regions as GeoJSON vector polygons
    regions_geojson_path = output_dir / "terrain_regions.geojson"
    try:
        _export_terrain_regions_geojson(regions_arr, dtm_clipped_path, regions_geojson_path)
        log.info(f"Terrain regions GeoJSON: {regions_geojson_path}")
    except Exception as e:
        log.warning(f"terrain_regions.geojson export failed (non-critical): {e}")

    # Step 8: Slope map image
    slope_map_path = output_dir / "slope_map.png"
    log.info("Rendering slope map...")
    _render_slope_map(slope_arr, regions_arr, slope_map_path)

    # Step 9: Compute 2K height mapping
    tk2_mapping = _compute_2k_height_mapping(z_min, z_max)

    # Assemble stats
    stats = {
        "z_min_m":              round(z_min, 2),
        "z_max_m":              round(z_max, 2),
        "z_range_m":            round(z_range, 2),
        "z_mean_m":             round(z_mean, 2),
        "resolution_m":         round(float(res_m), 2),
        "dtm_shape":            list(dtm_arr.shape),
        "slope_mean_deg":       round(float(np.nanmean(slope_arr)), 2),
        "slope_max_deg":        round(float(np.nanmax(slope_arr)), 2),
        "flat_pct":             round(float(np.sum(regions_arr == 1) / regions_arr.size * 100), 1),
        "gentle_pct":           round(float(np.sum(regions_arr == 2) / regions_arr.size * 100), 1),
        "moderate_pct":         round(float(np.sum(regions_arr == 3) / regions_arr.size * 100), 1),
        "steep_pct":            round(float(np.sum(regions_arr == 4) / regions_arr.size * 100), 1),
        "tk2_height_at_z_min":  tk2_mapping["min"],
        "tk2_height_at_z_max":  tk2_mapping["max"],
        "tk2_height_per_metre": tk2_mapping["per_metre"],
        "heightmap_path":       str(heightmap_path),
        "slope_map_path":       str(slope_map_path),
        "dtm_clipped_path":     str(dtm_clipped_path),
    }

    stats_path = output_dir / "terrain_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    log.info(f"Terrain stats written to {stats_path}")

    return stats


# ─── Reprojection ─────────────────────────────────────────────────────────────

def _reproject_to_itm(src_path: Path, dst_path: Path) -> None:
    """Reproject raster to ITM (EPSG:2157) at config.DTM_RESOLUTION_M."""
    dst_crs = CRS.from_epsg(2157)

    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        # Snap to target resolution
        res = config.DTM_RESOLUTION_M
        transform = rasterio.transform.from_origin(
            transform.c, transform.f,
            res, res,
        )
        width  = max(1, int(abs(src.bounds.right  - src.bounds.left)  / res) + 1)
        height = max(1, int(abs(src.bounds.top    - src.bounds.bottom) / res) + 1)

        kwargs = src.meta.copy()
        kwargs.update({
            "crs":       dst_crs,
            "transform": transform,
            "width":     width,
            "height":    height,
            "dtype":     "float32",
            "nodata":    -9999.0,
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )


def _clip_to_boundary_with_retry(
    dtm_path: Path, dst_path: Path,
    boundary_poly, boundary_data: dict
) -> None:
    """
    Clip DTM to course boundary with progressive buffer reduction.

    Problem 3: boundary buffers can exceed DEM coverage (especially for
    courses near Mapzen tile edges or coastal locations).

    Strategy: try buffers [300m, 150m, 50m, 0m] in order.
    Accept the first clip that has ≥5% valid coverage.
    """
    from pyproj import Transformer
    from shapely.ops import transform as shp_transform

    t = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
    boundary_itm = shp_transform(t.transform, boundary_poly)

    # Buffer schedule: generous first, then progressively tighter
    buffer_schedule = [config.BOUNDARY_BUFFER_M, 150, 50, 0]
    # Remove duplicates while preserving order
    seen = set()
    buffers = []
    for b in buffer_schedule:
        if b not in seen:
            seen.add(b)
            buffers.append(b)

    last_error = None
    for buf in buffers:
        try:
            clip_region = boundary_itm.buffer(buf) if buf > 0 else boundary_itm
            with rasterio.open(dtm_path) as src:
                out_image, out_transform = rio_mask(
                    src, [mapping(clip_region)], crop=True
                )
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver":    "GTiff",
                    "height":    out_image.shape[1],
                    "width":     out_image.shape[2],
                    "transform": out_transform,
                })

            # Quick validity check on clip result
            clip_arr = out_image[0].astype(np.float32)
            nodata_val = out_meta.get("nodata")
            if nodata_val is not None:
                valid_mask = clip_arr != nodata_val
            else:
                valid_mask = np.isfinite(clip_arr)

            valid_pct = float(valid_mask.sum()) / max(clip_arr.size, 1) * 100
            if valid_pct < 5.0:
                log.warning(
                    f"Clip with buffer={buf}m has only {valid_pct:.1f}% valid "
                    f"pixels — trying smaller buffer"
                )
                continue

            with rasterio.open(dst_path, "w", **out_meta) as dst:
                dst.write(out_image)

            log.info(f"Clip succeeded with buffer={buf}m ({valid_pct:.1f}% valid)")
            return

        except Exception as e:
            last_error = e
            log.warning(f"Clip failed with buffer={buf}m: {e}")
            continue

    raise RuntimeError(
        f"DEM clipping failed for all buffer values {buffers}. "
        f"The DEM may not cover the course boundary. Last error: {last_error}"
    )


# ─── Heightmap ────────────────────────────────────────────────────────────────

def _generate_heightmap(dtm_arr: np.ndarray, out_path: Path) -> None:
    """
    Export a 16-bit greyscale heightmap PNG normalised to 0–65535.
    NaN cells become 0 (sea level baseline).
    """
    z_min = float(np.nanmin(dtm_arr))
    z_max = float(np.nanmax(dtm_arr))
    z_range = z_max - z_min

    if z_range < 0.01:
        # Flat course — produce uniform mid-grey
        normalised = np.full(dtm_arr.shape, 32768, dtype=np.uint16)
    else:
        normalised = ((dtm_arr - z_min) / z_range * 65535)
        normalised = np.nan_to_num(normalised, nan=0.0)
        normalised = np.clip(normalised, 0, 65535).astype(np.uint16)

    # Resize to config.HEIGHTMAP_SIZE_PX × config.HEIGHTMAP_SIZE_PX
    img = Image.fromarray(normalised, mode="I;16")
    size = config.HEIGHTMAP_SIZE_PX
    img = img.resize((size, size), Image.LANCZOS)
    img.save(str(out_path))
    log.info(f"Heightmap saved: {out_path} ({size}×{size}px, 16-bit)")


# ─── V2: NaN fill and smoothing ──────────────────────────────────────────────

def _median_fill_nan(arr: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    V2: Fill NaN cells with local spatial median.
    Falls back to global median for isolated large nodata regions.
    Uses iterative passes so islands of NaN surrounded by valid data
    are filled before the global fallback is needed.
    """
    from scipy.ndimage import generic_filter

    if not np.any(np.isnan(arr)):
        return arr

    out = arr.copy()
    global_median = float(np.nanmedian(arr))

    # Up to 3 passes for NaN-surrounded cells
    for _ in range(3):
        if not np.any(np.isnan(out)):
            break
        nan_mask = np.isnan(out)

        def _local_median(values):
            valid = values[~np.isnan(values)]
            return float(np.median(valid)) if len(valid) > 0 else np.nan

        filled = generic_filter(
            out, _local_median,
            size=kernel, mode="nearest"
        )
        out[nan_mask] = filled[nan_mask]

    # Any remaining NaN → global median
    out = np.where(np.isnan(out), global_median, out)
    return out.astype(np.float32)


def _gaussian_smooth(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    V2: Apply Gaussian smoothing to reduce LiDAR/DEM point noise.
    sigma=1.0 is gentle — preserves macro terrain while killing sensor spikes.
    """
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma).astype(np.float32)


# ─── Slope ────────────────────────────────────────────────────────────────────

def _compute_slope(dtm_arr: np.ndarray, resolution_m: float) -> np.ndarray:
    """
    V2: NaN-safe slope computation using central difference (numpy gradient).
    Array must be pre-filled (no NaN) by _median_fill_nan before calling.
    """
    # Guard: if somehow NaN remain, fill with median
    if np.any(np.isnan(dtm_arr)):
        fill = float(np.nanmedian(dtm_arr))
        dtm_arr = np.where(np.isnan(dtm_arr), fill, dtm_arr)

    dy, dx = np.gradient(dtm_arr, resolution_m, resolution_m)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    return slope.astype(np.float32)


def _classify_terrain(slope_arr: np.ndarray) -> np.ndarray:
    """
    Classify terrain by slope into 4 regions:
      1 = flat     (< SLOPE_FLAT_MAX)
      2 = gentle   (SLOPE_FLAT_MAX – SLOPE_GENTLE_MAX)
      3 = moderate (SLOPE_GENTLE_MAX – SLOPE_MODERATE_MAX)
      4 = steep    (>= SLOPE_MODERATE_MAX)
    """
    regions = np.zeros(slope_arr.shape, dtype=np.uint8)
    regions[slope_arr <  config.SLOPE_FLAT_MAX]     = 1
    regions[(slope_arr >= config.SLOPE_FLAT_MAX)    & (slope_arr < config.SLOPE_GENTLE_MAX)]   = 2
    regions[(slope_arr >= config.SLOPE_GENTLE_MAX)  & (slope_arr < config.SLOPE_MODERATE_MAX)] = 3
    regions[slope_arr >= config.SLOPE_MODERATE_MAX] = 4
    regions[np.isnan(slope_arr)] = 0
    return regions


# ─── Slope map rendering ─────────────────────────────────────────────────────

def _render_slope_map(slope_arr: np.ndarray, regions_arr: np.ndarray, out_path: Path) -> None:
    """Render a colour-coded slope classification map and save as PNG."""
    # Colour map: 0=outside, 1=flat(green), 2=gentle(yellow), 3=moderate(orange), 4=steep(red)
    cmap = mcolors.ListedColormap(["#cccccc", "#4caf50", "#ffeb3b", "#ff9800", "#f44336"])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm   = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=config.FEATURE_MAP_DPI)
    ax.imshow(regions_arr, cmap=cmap, norm=norm, origin="upper")
    ax.set_axis_off()

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, fraction=0.03, pad=0.02,
        ticks=[0, 1, 2, 3, 4],
    )
    cbar.ax.set_yticklabels(["Outside", "Flat (<2°)", "Gentle (2–8°)", "Moderate (8–20°)", "Steep (>20°)"])

    plt.title("Terrain Slope Classification", pad=12)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=config.FEATURE_MAP_DPI)
    plt.close()
    log.info(f"Slope map saved: {out_path}")


# ─── 2K height mapping ────────────────────────────────────────────────────────

def _compute_2k_height_mapping(z_min: float, z_max: float) -> dict:
    """
    Compute the mapping from real-world elevation to 2K height slider (0–100).
    Returns per_metre value for use in build instructions.
    """
    z_range = max(z_max - z_min, 0.01)
    per_metre = (config.TK2_HEIGHT_MAX - config.TK2_HEIGHT_MIN) / z_range
    return {
        "min":       config.TK2_HEIGHT_MIN,
        "max":       config.TK2_HEIGHT_MAX,
        "per_metre": round(per_metre, 3),
        "z_min_m":   round(z_min, 2),
        "z_max_m":   round(z_max, 2),
    }


def z_to_2k(z_m: float, terrain_stats: dict) -> float:
    """Convert a real-world elevation (metres) to 2K height slider value (0–100)."""
    z_min = terrain_stats["z_min_m"]
    z_range = terrain_stats["z_range_m"]
    if z_range < 0.01:
        return 50.0
    val = (z_m - z_min) / z_range * config.TK2_HEIGHT_MAX
    return round(max(0.0, min(100.0, val)), 1)


# ─── V2: GeoJSON terrain regions export ──────────────────────────────────────

def _export_terrain_regions_geojson(
    regions_arr: np.ndarray,
    template_raster: Path,
    out_path: Path,
) -> None:
    """
    V2: Vectorise terrain_regions raster → GeoJSON polygons.
    Each feature has class (1–4) and label properties.
    """
    import json
    from rasterio.features import shapes
    from shapely.geometry import shape as shp_shape, mapping

    labels = {1: "flat", 2: "gentle", 3: "moderate", 4: "steep"}

    with rasterio.open(template_raster) as src:
        transform = src.transform
        crs = src.crs

    # Reproject to WGS84 for GeoJSON output
    t_to_wgs84 = Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
    from shapely.ops import transform as shp_transform

    features = []
    for region_class in [1, 2, 3, 4]:
        mask = (regions_arr == region_class).astype(np.uint8)
        for geom, val in shapes(mask, mask=mask, transform=transform):
            if val == 0:
                continue
            poly = shp_shape(geom)
            if poly.area < 1:
                continue
            try:
                poly_wgs84 = shp_transform(t_to_wgs84.transform, poly)
            except Exception:
                poly_wgs84 = poly
            features.append({
                "type": "Feature",
                "geometry": mapping(poly_wgs84),
                "properties": {
                    "class": int(region_class),
                    "label": labels.get(region_class, "unknown"),
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(geojson, indent=2))


# ─── Utility: write classified raster ────────────────────────────────────────

def _write_raster(arr: np.ndarray, out_path: Path, template_path: Path) -> None:
    """Write a uint8 array as GeoTIFF using metadata from template_path."""
    with rasterio.open(template_path) as src:
        meta = src.meta.copy()
    meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr[np.newaxis, :, :])

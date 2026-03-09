"""
vision_extract.py — Satellite imagery feature detection via computer vision.

UPGRADE 3 (V2.1): Detect golf features missing from OSM using satellite imagery.

Tile sources (in priority order):
  1. Bing Maps aerial tiles (best resolution for Ireland/UK)
  2. ESRI World Imagery tiles (global fallback)
  3. OpenStreetMap standard tiles (no satellite — last resort)

Detection targets:
  - fairways    (bright green elongated shapes)
  - greens      (small circular uniform patches)
  - bunkers     (light sand-coloured irregular shapes)
  - water       (dark blue/grey regions)
  - trees       (dark textured green clusters)
  - rough       (medium-texture yellowy-green zones)

Libraries: opencv-python, scikit-image, numpy, Pillow, mercantile
Output:    GeoJSON per feature type + debug composite image
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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


# ─── Satellite tile sources ───────────────────────────────────────────────────

TILE_SOURCES = [
    # Bing Aerial — high quality for Ireland/UK; no key for basic usage
    {
        "name":     "Bing",
        "url":      "https://t.ssl.ak.dynamic.tiles.virtualearth.net/comp/ch/{quadkey}?mkt=en-IE&it=A&shading=hill&og=2177&n=z",
        "type":     "quadkey",
        "max_zoom": 19,
    },
    # ESRI World Imagery — global, no key required
    {
        "name":  "ESRI",
        "url":   "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "type":  "xyz",
        "max_zoom": 18,
    },
    # Stamen/CARTO topo (no satellite, last resort for terrain context only)
    {
        "name":  "OSM",
        "url":   "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "type":  "xyz",
        "max_zoom": 19,
    },
]


# ─── HSV colour thresholds (OpenCV: H 0-179, S 0-255, V 0-255) ───────────────

THRESHOLDS = {
    "fairway": {
        "h": (35, 85),   # green-yellow hue
        "s": (40, 220),  # moderately saturated
        "v": (50, 200),  # not too dark, not bleached
        "min_area_px": 500,
        "texture_max": 35,   # fairways are uniform
    },
    "green": {
        "h": (38, 82),   # brighter green
        "s": (50, 210),
        "v": (60, 195),
        "min_area_px": 80,
        "max_area_px": 400,
        "circularity_min": 0.50,  # putting greens are roundish (tighter)
        "texture_max": 30,        # very uniform surface
    },
    "bunker": {
        "h": (14, 42),   # pale yellow → beige → tan
        "s": (15, 110),  # low saturation (sand)
        "v": (155, 255), # bright (sand reflects light)
        "min_area_px": 80,
        "max_area_px": 600,
    },
    "water": {
        "h": (90, 140),  # blue-cyan
        "s": (30, 230),
        "v": (0, 140),   # water is typically dark
        "min_area_px": 50,
        # Also catch very dark regions (deep shadow / dark water)
        "dark_v_max": 45,   # if V < 45 and H in range, always water
    },
    "trees": {
        "h": (30, 85),   # green hues
        "s": (25, 200),
        "v": (15, 110),  # dark — trees absorb light
        "min_area_px": 100,
        "texture_min": 28,   # trees are textured (unlike smooth fairways)
    },
    "rough": {
        "h": (28, 88),   # broad green range
        "s": (20, 180),
        "v": (40, 160),
        "min_area_px": 300,
        "texture_range": (12, 45),  # medium texture (not as smooth as fairway)
    },
}


# ─── Geometric filter constants ───────────────────────────────────────────────
# These control false-positive suppression. Increase to reduce detections,
# decrease to recover more features. Values are in pixels² or ratios.

# Bunkers: small-to-medium irregular sand patches
MIN_BUNKER_AREA_PX  = 80    # ignore tiny noise (was 20)
MAX_BUNKER_AREA_PX  = 600   # ignore oversized patches/paths (was 1500)
MAX_BUNKER_ASPECT   = 4.0   # bounding-box long/short ratio — bunkers aren't thin lines

# Greens: small, compact, circular putting surfaces
MIN_GREEN_AREA_PX   = 60    # minimum size
MAX_GREEN_AREA_PX   = 600   # maximum size (relaxed from 400 — some greens appear larger at zoom 16)
MIN_GREEN_CIRC      = 0.38  # circularity threshold (relaxed from 0.50 — not all greens are round)

# Fairways: large, elongated corridors
MIN_FAIRWAY_AREA_PX = 500   # must be a substantial patch (was 400)
MIN_FAIRWAY_ASPECT  = 1.5   # must be elongated, not a circular blob


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_features(
    bbox_wgs84: list,
    output_dir: Path,
    zoom: int = 17,
    boundary_data: Optional[dict] = None,
) -> dict:
    """
    Download satellite tiles and detect golf features using computer vision.

    UPGRADE 1: When boundary_data is provided, a course boundary raster mask
    is applied to every detection pass, eliminating features outside the
    golf course polygon.

    Args:
        bbox_wgs84:     [min_lon, min_lat, max_lon, max_lat]
        output_dir:     Directory for output GeoJSON and debug images
        zoom:           Tile zoom level (17 ≈ 1.2m/px, 16 ≈ 2.4m/px)
        boundary_data:  Optional boundary dict from boundary.resolve_boundary()
                        When provided, all detections are clipped to the
                        course polygon.

    Returns:
        Detection summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Vision extraction: bbox={bbox_wgs84}, zoom={zoom}")

    # Step 1: Download satellite mosaic
    try:
        img_arr, transform_params = _download_satellite_mosaic(bbox_wgs84, zoom, output_dir)
    except Exception as e:
        log.warning(f"Satellite tile download failed: {e}")
        return {"error": str(e), "detections": {}}

    if img_arr is None:
        return {"detections": {}}

    log.info(f"Satellite mosaic: {img_arr.shape[1]}×{img_arr.shape[0]}px")

    # Save mosaic for reference
    try:
        _save_debug_image(img_arr, output_dir / "satellite_mosaic.jpg")
    except Exception:
        pass

    # UPGRADE 1: Build course boundary raster mask ────────────────────────────
    # All detections will be filtered against this mask before saving so that
    # no feature can exist outside the golf course polygon.
    course_mask    = None
    course_polygon = None
    if boundary_data:
        try:
            from pipeline.course_mask import (
                load_course_polygon,
                build_raster_mask,
            )
            course_polygon = load_course_polygon(boundary_data)
            course_mask    = build_raster_mask(
                course_polygon, transform_params, img_arr.shape
            )
        except Exception as e:
            log.warning(f"Course mask setup failed (continuing without mask): {e}")

    # Step 2: Run all detectors
    detections = {}
    detection_fns = [
        ("fairway", _detect_fairways),
        ("green",   _detect_greens),
        ("bunker",  _detect_bunkers),
        ("water",   _detect_water),
        ("trees",   _detect_trees),
        ("rough",   _detect_rough),
    ]

    all_masks = {}  # for debug overlay
    class_filter_stats = {}  # per-stage counts for class_filter_debug.json

    for feat_type, fn in detection_fns:
        try:
            contours, mask = fn(img_arr)
            raw_count = len(contours)

            # UPGRADE 1 (pixel level): filter contours to course boundary
            after_boundary = raw_count
            if course_mask is not None:
                try:
                    from pipeline.course_mask import apply_mask_to_contours
                    before = len(contours)
                    contours = apply_mask_to_contours(contours, course_mask)
                    after_boundary = len(contours)
                    if after_boundary < before:
                        log.debug(
                            f"Vision [{feat_type}]: boundary mask removed "
                            f"{before - after_boundary} out-of-bounds contours"
                        )
                except Exception as me:
                    log.debug(f"Mask application failed for {feat_type}: {me}")

            all_masks[feat_type] = mask
            out_path = output_dir / f"vision_{feat_type}s.geojson"
            _save_detections_geojson(
                contours, transform_params, feat_type, out_path,
                course_polygon=course_polygon,
            )
            # Count from saved file (reflects GeoJSON-level clip)
            try:
                saved = json.loads(out_path.read_text(encoding="utf-8"))
                saved_count = len(saved.get("features", []))
            except Exception:
                saved_count = len(contours)

            detections[feat_type] = {
                "count": saved_count,
                "path":  str(out_path),
            }
            class_filter_stats[feat_type] = {
                "raw_contours":        raw_count,
                "after_boundary_mask": after_boundary,
                "after_area_clip":     saved_count,
            }
            log.info(
                f"Vision [{feat_type}]: {raw_count} raw → "
                f"{after_boundary} in-bounds → {saved_count} after m² filter"
            )
        except Exception as e:
            log.warning(f"Vision [{feat_type}] detection failed: {e}")

    # Write per-stage count debug artifact
    try:
        (output_dir / "class_filter_debug.json").write_text(
            json.dumps({"pass": 1, "feature_counts": class_filter_stats}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Step 3: Save debug overlay with all detections coloured
    try:
        _save_detection_overlay(img_arr, all_masks, output_dir / "vision_overlay.jpg")
    except Exception as e:
        log.debug(f"Overlay save failed: {e}")

    summary = {
        "zoom":             zoom,
        "image_size":       [img_arr.shape[1], img_arr.shape[0]],
        "transform_params": list(transform_params),  # (west, north, lon/px, lat/px)
        "detections":       detections,
    }
    (output_dir / "vision_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def refilter_water_strict(output_dir: Path) -> int:
    """
    Post-hoc strict water filter.

    Called by run_pipeline when water count > 25 after the main detection
    pass.  Re-reads vision_waters.geojson, applies stricter real-world m²
    area and compactness filters, then overwrites the file in-place.

    Thresholds:
      - Minimum area: 2000 m² (≈ 50×40m pond — smaller = drainage noise)
      - Compactness ≥ 0.06 (rejects thin linear shadows)

    Returns new water feature count (or 0 on error).
    """
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    path = Path(output_dir) / "vision_waters.geojson"
    if not path.exists():
        return 0

    try:
        data     = json.loads(path.read_text(encoding="utf-8"))
        features = data.get("features", [])
        if not features:
            return 0

        t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        MIN_WATER_M2_STRICT  = 2000   # m²: genuine golf water bodies are large
        MIN_WATER_COMPACT    = 0.06   # very thin = road shadow, not pond

        filtered = []
        for f in features:
            geom_dict = f.get("geometry")
            if not geom_dict:
                continue
            try:
                poly     = shape(geom_dict)
                poly_itm = shp_transform(t_to_itm.transform, poly)
                area_m2  = poly_itm.area
                if area_m2 < MIN_WATER_M2_STRICT:
                    continue
                perimeter = poly_itm.length
                if perimeter > 0:
                    compactness = 4 * np.pi * area_m2 / (perimeter ** 2)
                    if compactness < MIN_WATER_COMPACT:
                        continue
                filtered.append(f)
            except Exception:
                continue

        _write_geojson_features(filtered, path)
        log.info(
            f"Water strict refilter: {len(features)} → {len(filtered)} "
            f"(≥2000m², compactness ≥{MIN_WATER_COMPACT})"
        )
        return len(filtered)

    except Exception as e:
        log.warning(f"refilter_water_strict failed: {e}")
        return 0


def merge_vision_with_osm(
    vision_summary: dict,
    osm_output_dir: Path,
    min_iou_threshold: float = 0.25,
) -> dict:
    """
    Merge vision detections with existing OSM GeoJSON features.
    Adds vision features that have low IoU overlap with OSM features.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    stats = {}
    feat_type_to_file = {
        "fairway": "fairways.geojson",
        "green":   "greens.geojson",
        "bunker":  "bunkers.geojson",
        "water":   "water.geojson",
        "trees":   "trees.geojson",
        "rough":   "rough.geojson",
    }

    for feat_type, det_info in vision_summary.get("detections", {}).items():
        vision_path = Path(det_info.get("path", ""))
        osm_filename = feat_type_to_file.get(feat_type, f"{feat_type}s.geojson")
        osm_path = osm_output_dir / osm_filename

        if not vision_path.exists():
            continue

        vision_feats = _load_geojson(vision_path)
        osm_feats = _load_geojson(osm_path) if osm_path.exists() else []

        osm_geoms = []
        for f in osm_feats:
            if f.get("geometry"):
                try:
                    osm_geoms.append(shape(f["geometry"]))
                except Exception:
                    pass
        osm_union = unary_union(osm_geoms) if osm_geoms else None

        added = []
        for vf in vision_feats:
            if not vf.get("geometry"):
                continue
            try:
                vg = shape(vf["geometry"])
            except Exception:
                continue
            if osm_union is None or not osm_union.intersects(vg):
                vf["properties"]["source"] = "vision"
                added.append(vf)
            else:
                try:
                    inter_area = osm_union.intersection(vg).area
                    union_area = osm_union.area + vg.area - inter_area
                    iou = inter_area / max(union_area, 1e-10)
                    if iou < min_iou_threshold:
                        vf["properties"]["source"] = "vision"
                        added.append(vf)
                except Exception:
                    pass

        if added:
            # Step 7: Multi-source fusion — merge overlapping geometries rather
            # than blindly appending.  For each vision feature that partially
            # overlaps an OSM feature, union the two geometries into one.
            fused_added = []
            for vf in added:
                try:
                    vg = shape(vf["geometry"])
                    merged_with_osm = False
                    for idx, of in enumerate(osm_feats):
                        if not of.get("geometry"):
                            continue
                        og = shape(of["geometry"])
                        if og.intersects(vg):
                            try:
                                union_geom = og.union(vg)
                                if union_geom.is_valid and not union_geom.is_empty:
                                    from shapely.geometry import mapping
                                    osm_feats[idx]["geometry"] = mapping(union_geom)
                                    osm_feats[idx]["properties"]["source"] = "osm+vision"
                                    merged_with_osm = True
                                    break
                            except Exception:
                                pass
                    if not merged_with_osm:
                        fused_added.append(vf)
                except Exception:
                    fused_added.append(vf)

            merged_feats = osm_feats + fused_added
            _write_geojson_features(merged_feats, osm_path)
            log.info(
                f"Vision merge [{feat_type}]: +{len(fused_added)} new, "
                f"{len(added) - len(fused_added)} merged with OSM → {osm_path.name}"
            )

        stats[feat_type] = len(added)

    return stats


# ─── Satellite tile download ──────────────────────────────────────────────────

def _download_satellite_mosaic(
    bbox_wgs84: list,
    zoom: int,
    output_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Download and stitch satellite tiles.

    Tries TILE_SOURCES in order. Returns (rgb_array, transform_params) or (None, None).
    transform_params: (west_lon, north_lat, lon_per_px, lat_per_px)
    """
    try:
        import mercantile
        from PIL import Image
        import io as _io
        import requests
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}")

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))

    # Auto step-down if tile count is unmanageable
    if len(tiles) > 64 and zoom > 13:
        log.info(f"Too many tiles ({len(tiles)}) at zoom {zoom} — stepping down")
        return _download_satellite_mosaic(bbox_wgs84, zoom - 1, output_dir)

    if not tiles:
        return None, None

    log.info(f"Downloading {len(tiles)} satellite tiles at zoom {zoom}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "CourseReplicator2K/2.0 (research)",
        "Referer":    "https://github.com/",
    })

    # Try each tile source in order
    for source in TILE_SOURCES:
        tile_images = _fetch_tiles_from_source(source, tiles, session)
        if len(tile_images) >= max(1, len(tiles) // 2):
            log.info(f"Using tile source: {source['name']} ({len(tile_images)}/{len(tiles)} tiles)")
            break
    else:
        return None, None

    # Stitch mosaic
    all_tile_objs = [t for _, t in tile_images.values()]
    x_coords = [t.x for t in all_tile_objs]
    y_coords = [t.y for t in all_tile_objs]
    min_tx, max_tx = min(x_coords), max(x_coords)
    min_ty, max_ty = min(y_coords), max(y_coords)

    tw, th = 256, 256
    grid_w = (max_tx - min_tx + 1) * tw
    grid_h = (max_ty - min_ty + 1) * th
    mosaic = Image.new("RGB", (grid_w, grid_h), color=(100, 100, 100))

    for (tx, ty), (img, tile) in tile_images.items():
        px = (tx - min_tx) * tw
        py = (ty - min_ty) * th
        mosaic.paste(img, (px, py))

    # Compute geographic transform
    ul_bounds = mercantile.bounds(mercantile.Tile(min_tx, min_ty, zoom))
    lr_bounds = mercantile.bounds(mercantile.Tile(max_tx, max_ty, zoom))
    total_west  = ul_bounds.west
    total_north = ul_bounds.north
    lon_per_px  = (lr_bounds.east  - ul_bounds.west)  / grid_w
    lat_per_px  = (ul_bounds.north - lr_bounds.south) / grid_h
    transform_params = (total_west, total_north, lon_per_px, lat_per_px)

    return np.array(mosaic, dtype=np.uint8), transform_params


def _fetch_tiles_from_source(source: dict, tiles: list, session) -> dict:
    """
    Fetch tiles from one source definition.
    Returns dict: (tx, ty) → (PIL.Image, mercantile.Tile)
    """
    from PIL import Image
    import io as _io

    results = {}
    url_template = source["url"]
    source_type  = source["type"]

    for tile in tiles:
        try:
            if source_type == "quadkey":
                qk = _tile_to_quadkey(tile.x, tile.y, tile.z)
                url = url_template.format(quadkey=qk)
            else:
                url = url_template.format(z=tile.z, x=tile.x, y=tile.y)

            resp = session.get(url, timeout=12)
            if resp.status_code != 200:
                continue

            ct = resp.headers.get("Content-Type", "")
            if "image" not in ct and "octet" not in ct:
                continue

            img = Image.open(_io.BytesIO(resp.content)).convert("RGB")
            results[(tile.x, tile.y)] = (img, tile)

        except Exception as e:
            log.debug(f"Tile {tile} from {source['name']} failed: {e}")
            continue

    return results


def _tile_to_quadkey(x: int, y: int, z: int) -> str:
    """Convert tile (x, y, z) to Bing Maps quadkey string."""
    quadkey = []
    for i in range(z, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if x & mask:
            digit += 1
        if y & mask:
            digit += 2
        quadkey.append(str(digit))
    return "".join(quadkey)


# ─── Feature detectors ────────────────────────────────────────────────────────

def _get_cv2():
    """Import cv2 with a helpful error message."""
    try:
        import cv2
        return cv2
    except ImportError:
        raise RuntimeError(
            "opencv-python is required for vision extraction. "
            "Install with: pip install opencv-python"
        )


def _hsv_mask(img_rgb: np.ndarray, h_range, s_range, v_range) -> np.ndarray:
    """Create a binary mask from HSV threshold ranges."""
    cv2 = _get_cv2()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lo = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
    hi = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _local_texture(img_rgb: np.ndarray, kernel: int = 7) -> np.ndarray:
    """
    Compute local texture as standard deviation of V channel.
    High stddev = textured (trees, rough); low stddev = uniform (fairway, green).
    """
    cv2 = _get_cv2()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v   = hsv[:, :, 2].astype(np.float32)
    # Local std via morphological approximation
    v_blur = cv2.blur(v, (kernel, kernel))
    v_sq   = cv2.blur(v ** 2, (kernel, kernel))
    variance = np.maximum(v_sq - v_blur ** 2, 0)
    return np.sqrt(variance)


def _morph_clean(mask: np.ndarray, open_k: int = 5, close_k: int = 7) -> np.ndarray:
    """Apply morphological open (remove noise) then close (fill gaps)."""
    cv2 = _get_cv2()
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open,  iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close, iterations=2)
    return mask


def _detect_fairways(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect fairway regions: bright green, elongated, uniform texture.

    UPGRADE 8 — Fairway continuity:
      After standard detection, apply a larger morphological close (15px, 3x)
      to merge nearby fragmented segments.  Dogleg fairways that produce two
      disconnected polygons are re-joined into one contiguous shape.
    """
    cv2 = _get_cv2()
    t = THRESHOLDS["fairway"]

    mask = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture = _local_texture(img_rgb)
    # Fairways are smooth — exclude high-texture areas (trees/rough)
    mask[texture > t["texture_max"]] = 0
    mask = _morph_clean(mask, open_k=7, close_k=9)

    # Extra large closing pass to merge fragmented dogleg segments
    k_continuity = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_continuity, iterations=3)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_FAIRWAY_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect < MIN_FAIRWAY_ASPECT:
            continue
        valid.append(c)
    return valid, mask


def _detect_greens(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect putting greens: small, circular, very uniform green patches.
    Uses combined HSV mask + circularity + size + texture filters.
    Geometric filters: MIN/MAX_GREEN_AREA_PX, MIN_GREEN_CIRC.
    """
    cv2 = _get_cv2()
    t = THRESHOLDS["green"]

    mask = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture = _local_texture(img_rgb)
    mask[texture > t["texture_max"]] = 0
    mask = _morph_clean(mask, open_k=3, close_k=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_GREEN_AREA_PX or area > MAX_GREEN_AREA_PX:
            continue
        perim = cv2.arcLength(c, True)
        if perim < 1:
            continue
        circularity = 4 * np.pi * area / (perim ** 2)
        if circularity < MIN_GREEN_CIRC:
            continue
        candidates.append(c)

    return candidates, mask


def _detect_bunkers(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect bunkers: light sand-coloured bright irregular patches.

    UPGRADE 6 — Improved sand HSV signature:
      H: 14–45  (pale yellow → beige → tan sand tones)
      S: 10–120 (low saturation — dry sand is almost white)
      V: 155–255 (high brightness — sand reflects strongly)

    Rejects: thin lines (roads, paths), rooftop edges (aspect ratio cap).
    """
    cv2 = _get_cv2()

    # Improved sand HSV (broader S range, narrower H to avoid roads)
    mask = _hsv_mask(img_rgb, (14, 45), (10, 120), (155, 255))
    mask = _morph_clean(mask, open_k=3, close_k=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_BUNKER_AREA_PX <= area <= MAX_BUNKER_AREA_PX):
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > MAX_BUNKER_ASPECT:
            continue
        valid.append(c)
    return valid, mask


def _detect_water(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect water hazards.

    COUNCIL REDESIGN — Stricter multi-signal water detection.

    Combines four independent signals; a pixel is marked as water only
    when AT LEAST 3 of the 4 signals agree (was 2 — caused 173 false
    positives on Old Conna due to road shadows and dark tree patches).

    Signals:
      1. NDWI approx  (Green-Red)/(Green+Red) > 0.05
      2. Blue dominance  B > R×1.15  and  B > G×0.85  and  B > 60
      3. HSV blue-hue mask  (tighter: H 95-135, dark V)
      4. Dark-value mask  (V < 40), excluding green and yellow hues

    Post-contour shape filters (conservative — false negatives preferred):
      - Minimum area: 1000 px (was 200) — eliminates drainage noise
      - Maximum aspect ratio: 8.0 — rejects road/path shadows
      - Minimum compactness: 0.06 — rejects linear shadows

    Logs: "Water raw: X contours → after shape filter: Y"
    """
    cv2 = _get_cv2()
    t   = THRESHOLDS["water"]

    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    # Signal 1: NDWI approximation (stricter threshold)
    ndwi         = (g - r) / (g + r + 1e-6)
    sig_ndwi     = (ndwi > 0.05).astype(np.uint8)

    # Signal 2: Blue channel dominance (stricter ratios — was 1.1/0.9/40)
    sig_blue_dom = (
        (b > r * 1.15) & (b > g * 0.85) & (b > 60)
    ).astype(np.uint8)

    # Signal 3: HSV blue/cyan hue (tighter range)
    mask_hsv = _hsv_mask(img_rgb, (95, 135), (40, 230), (0, 130))
    sig_hsv  = (mask_hsv > 0).astype(np.uint8)

    # Signal 4: Dark regions (V < 40), exclude green AND yellow hues
    hsv      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    dark_v   = (hsv[:, :, 2] < 40).astype(np.uint8)
    is_green_yellow = (hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 95)
    dark_v[is_green_yellow] = 0
    sig_dark = dark_v

    # Require ≥3 signals (was ≥2 — the cause of 173 false positives)
    signal_sum = sig_ndwi + sig_blue_dom + sig_hsv + sig_dark
    mask = (signal_sum >= 3).astype(np.uint8) * 255

    mask = _morph_clean(mask, open_k=7, close_k=11)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_count = len(contours)

    # Shape filters — water is conservative: false negatives preferred
    MIN_WATER_PX     = 1000   # was 200; real water bodies are large
    MAX_WATER_ASPECT = 8.0    # super-elongated = road shadow, not pond
    MIN_COMPACTNESS  = 0.06   # very thin = drainage ditch, not hazard

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_WATER_PX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > MAX_WATER_ASPECT:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
            if compactness < MIN_COMPACTNESS:
                continue
        valid.append(c)

    log.info(
        f"Water detection: {raw_count} raw contours → "
        f"{len(valid)} after shape filter (min {MIN_WATER_PX}px, "
        f"aspect ≤{MAX_WATER_ASPECT}, compactness ≥{MIN_COMPACTNESS})"
    )
    return valid, mask


def _detect_trees(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect tree clusters: dark green, high texture.
    Trees appear as dark, rough-textured green masses in aerial imagery.
    """
    cv2 = _get_cv2()
    t = THRESHOLDS["trees"]

    mask = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture = _local_texture(img_rgb)
    # Trees MUST be textured — exclude smooth areas (fairways, greens)
    mask[texture < t["texture_min"]] = 0
    mask = _morph_clean(mask, open_k=5, close_k=9)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= t["min_area_px"]]
    return valid, mask


def _detect_rough(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Detect rough zones: medium-texture green/yellow-green areas.
    Rough is less uniform than fairways but less dark/textured than trees.
    """
    cv2 = _get_cv2()
    t = THRESHOLDS["rough"]
    tex_lo, tex_hi = t["texture_range"]

    mask = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture = _local_texture(img_rgb)
    # Rough has medium texture
    tex_mask = ((texture >= tex_lo) & (texture <= tex_hi)).astype(np.uint8) * 255
    mask = cv2.bitwise_and(mask, tex_mask)
    mask = _morph_clean(mask, open_k=7, close_k=11)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= t["min_area_px"]]
    return valid, mask


# ─── GeoJSON conversion ───────────────────────────────────────────────────────

def _save_detections_geojson(
    contours: list,
    transform_params: Optional[tuple],
    feature_type: str,
    out_path: Path,
    course_polygon=None,
) -> None:
    """
    Convert pixel contours → WGS84 GeoJSON polygons, apply real-world m²
    area filters from config, optionally clip to course boundary, and save.

    UPGRADE 1 (GeoJSON level): when course_polygon is provided, every
    feature is clipped to the boundary polygon as a final clean-up step
    after pixel-level masking.

    Real-world filter thresholds (config.py):
      bunker:  MIN_BUNKER_AREA_M2 – MAX_BUNKER_AREA_M2
      green:   MIN_GREEN_AREA_M2  – MAX_GREEN_AREA_M2
      fairway: MIN_FAIRWAY_AREA_M2+
    """
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer
    cv2 = _get_cv2()

    if transform_params is None:
        _write_geojson_features([], out_path)
        return

    west, north, lon_per_px, lat_per_px = transform_params
    features = []

    # Build a WGS84→ITM projector for metric area calculation
    try:
        t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    except Exception:
        t_to_itm = None

    # Real-world area bounds per feature type (m²) — from config
    area_bounds = {
        "bunker":  (config.MIN_BUNKER_AREA_M2,  config.MAX_BUNKER_AREA_M2),
        "green":   (config.MIN_GREEN_AREA_M2,   config.MAX_GREEN_AREA_M2),
        "fairway": (config.MIN_FAIRWAY_AREA_M2, None),
    }
    min_m2, max_m2 = area_bounds.get(feature_type, (None, None))

    total_in  = len(contours)
    total_out = 0

    for contour in contours:
        pts = contour.squeeze()
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue

        coords = [
            (west + float(px) * lon_per_px,
             north - float(py) * lat_per_px)
            for px, py in pts
        ]
        if len(coords) < 3:
            continue

        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty or poly.area < 1e-12:
                continue

            # Real-world m² area filter
            if (min_m2 is not None or max_m2 is not None) and t_to_itm is not None:
                try:
                    poly_itm  = shp_transform(t_to_itm.transform, poly)
                    area_m2   = poly_itm.area
                    if min_m2 is not None and area_m2 < min_m2:
                        continue
                    if max_m2 is not None and area_m2 > max_m2:
                        continue
                except Exception:
                    pass  # if projection fails, keep the feature

            # Confidence score: higher for features in the expected size range
            confidence = _vision_confidence(feature_type, poly, t_to_itm)

            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "type":       feature_type,
                    "source":     "vision",
                    "area_px":    float(cv2.contourArea(contour)),
                    "confidence": confidence,
                },
            })
            total_out += 1
        except Exception:
            continue

    if total_in != total_out:
        log.info(f"Vision filter applied — {total_in} → {total_out} {feature_type}s retained")

    # UPGRADE 1 (GeoJSON level): final boundary clip
    if course_polygon is not None and features:
        try:
            from pipeline.course_mask import clip_geojson_to_boundary
            features = clip_geojson_to_boundary(features, course_polygon)
        except Exception as e:
            log.debug(f"GeoJSON boundary clip failed for {feature_type}: {e}")

    _write_geojson_features(features, out_path)


def _vision_confidence(
    feature_type: str,
    poly_wgs84,
    t_to_itm,
) -> float:
    """
    Compute a confidence score (0.0–1.0) for a vision-detected feature.

    Considers:
      - Whether the feature area is within the expected range (config)
      - Shape validity
    """
    base = 0.55  # vision detections start at MEDIUM confidence

    area_bounds = {
        "bunker":  (config.MIN_BUNKER_AREA_M2,  config.MAX_BUNKER_AREA_M2),
        "green":   (config.MIN_GREEN_AREA_M2,   config.MAX_GREEN_AREA_M2),
        "fairway": (config.MIN_FAIRWAY_AREA_M2, None),
    }
    min_m2, max_m2 = area_bounds.get(feature_type, (None, None))

    if t_to_itm is not None and (min_m2 or max_m2):
        try:
            from shapely.ops import transform as shp_transform
            poly_itm = shp_transform(t_to_itm.transform, poly_wgs84)
            area_m2  = poly_itm.area
            lo = min_m2 or 0
            hi = max_m2 or float("inf")
            if lo <= area_m2 <= hi:
                base += 0.15   # in expected range → boost
            elif area_m2 < lo * 0.5 or (max_m2 and area_m2 > hi * 2):
                base -= 0.15   # very far out of range → penalise
        except Exception:
            pass

    if not poly_wgs84.is_valid:
        base -= 0.10

    return round(max(0.0, min(1.0, base)), 2)


# ─── Debug output ─────────────────────────────────────────────────────────────

OVERLAY_COLOURS_BGR = {
    "fairway": (100, 220, 100),   # green
    "green":   (0,   180,   0),   # dark green
    "bunker":  (100, 210, 255),   # sand/yellow
    "water":   (220,  50,  50),   # blue
    "trees":   (0,    80,   0),   # very dark green
    "rough":   (60,  130,  60),   # olive
}


def _save_debug_image(img_rgb: np.ndarray, out_path: Path) -> None:
    from PIL import Image
    Image.fromarray(img_rgb).save(str(out_path), quality=82)


def _save_detection_overlay(
    img_rgb: np.ndarray,
    all_masks: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Save satellite mosaic with semi-transparent detection overlays."""
    cv2 = _get_cv2()

    overlay = img_rgb.copy()
    colour_map = {
        "fairway": (100, 220, 100),
        "green":   (0,   200,   0),
        "bunker":  (255, 220, 100),
        "water":   (50,   50, 220),
        "trees":   (0,    60,   0),
        "rough":   (80,  140,  80),
    }

    for feat_type, mask in all_masks.items():
        colour = colour_map.get(feat_type, (128, 128, 128))
        coloured = np.zeros_like(img_rgb)
        coloured[mask > 0] = colour
        overlay = cv2.addWeighted(overlay, 0.75, coloured, 0.25, 0)

    from PIL import Image
    Image.fromarray(overlay).save(str(out_path), quality=82)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _write_geojson_features(features: list, out_path: Path) -> None:
    out_path.write_text(json.dumps(
        {"type": "FeatureCollection", "features": features}, indent=2
    ), encoding="utf-8")


def _load_geojson(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("features", [])
    except Exception:
        return []

"""
vision_extract.py — Satellite imagery download + golf feature detection.

Step 2 — Multi-provider satellite tile system (priority order):
  1. Google Satellite     — https://mt1.google.com/vt/lyrs=s
  2. ESRI World Imagery   — ArcGIS Online, global, no key
  3. Bing Aerial          — best quality for Ireland/UK

Tiles are stitched into a SATELLITE_MOSAIC_SIZE × SATELLITE_MOSAIC_SIZE mosaic
(default 2048×2048) and saved as satellite_mosaic.jpg.

Step 5 — HSV feature detection:
  fairways  — medium/bright green elongated corridors
  greens    — small, circular, high-V green patches
  bunkers   — pale yellow/sand, high V, low S
  water     — blue/cyan, multi-signal (NDWI + hue + dark-V)

Output:
  vision_fairways.geojson
  vision_greens.geojson
  vision_bunkers.geojson
  vision_waters.geojson
  satellite_mosaic.jpg
  vision_debug.png           ← Step 9 debug overlay
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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


# ─── Satellite tile source registry ───────────────────────────────────────────
# Each entry:  name, url template, type (xyz | quadkey), max_zoom

def _build_tile_sources() -> List[Dict]:
    """Build ordered tile source list from config.SATELLITE_SOURCES."""
    _all = {
        "google": {
            "name":     "Google Satellite",
            "url":      getattr(config, "GOOGLE_TILE_URL",
                                "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"),
            "type":     "xyz",
            "max_zoom": 20,
            "headers":  {
                "Referer": "https://maps.google.com",
                "User-Agent": "Mozilla/5.0 CourseReplicator2K",
            },
        },
        "esri": {
            "name":    "ESRI World Imagery",
            "url":     getattr(config, "ESRI_TILE_URL",
                               "https://services.arcgisonline.com/ArcGIS/rest/services/"
                               "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
            "type":    "xyz",
            "max_zoom": 19,
            "headers": {"User-Agent": "CourseReplicator2K/2.0"},
        },
        "bing": {
            "name":    "Bing Aerial",
            "url":     getattr(config, "BING_TILE_URL",
                               "https://t.ssl.ak.dynamic.tiles.virtualearth.net/comp/ch/{quadkey}"
                               "?mkt=en-IE&it=A&shading=hill&og=2177&n=z"),
            "type":    "quadkey",
            "max_zoom": 19,
            "headers": {"User-Agent": "CourseReplicator2K/2.0"},
        },
    }
    order = getattr(config, "SATELLITE_SOURCES", ["google", "esri", "bing"])
    return [_all[k] for k in order if k in _all]


TILE_SOURCES = _build_tile_sources()


# ─── HSV colour thresholds ────────────────────────────────────────────────────
# OpenCV convention: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255]
# Values tuned for satellite imagery at zoom 16–18.

THRESHOLDS = {
    "fairway": {
        "h":           (35, 85),
        "s":           (40, 220),
        "v":           (50, 200),
        "min_area_px": 500,
        "texture_max": 35,
    },
    "green": {
        "h":              (38, 82),
        "s":              (50, 210),
        "v":              (60, 195),
        "min_area_px":    80,
        "max_area_px":    400,
        "circularity_min": 0.50,
        "texture_max":    30,
    },
    "bunker": {
        "h":           (14, 42),
        "s":           (15, 110),
        "v":           (155, 255),
        "min_area_px": 80,
        "max_area_px": 600,
    },
    "water": {
        "h":           (90, 140),
        "s":           (30, 230),
        "v":           (0, 140),
        "min_area_px": 50,
        "dark_v_max":  45,
    },
    "trees": {
        "h":           (30, 85),
        "s":           (25, 200),
        "v":           (15, 110),
        "min_area_px": 100,
        "texture_min": 28,
    },
    "rough": {
        "h":             (28, 88),
        "s":             (20, 180),
        "v":             (40, 160),
        "min_area_px":   300,
        "texture_range": (12, 45),
    },
}

# Geometric filter constants (pixels²)
MIN_BUNKER_AREA_PX  = 80
MAX_BUNKER_AREA_PX  = 600
MAX_BUNKER_ASPECT   = 4.0

MIN_GREEN_AREA_PX   = 60
MAX_GREEN_AREA_PX   = 600
MIN_GREEN_CIRC      = 0.38

MIN_FAIRWAY_AREA_PX = 500
MIN_FAIRWAY_ASPECT  = 1.5

# Debug overlay colours (RGB)
_DBG_RGB = {
    "fairway": (50, 180, 50),
    "green":   (0,  255, 80),
    "bunker":  (255, 200, 0),
    "water":   (30, 100, 255),
    "trees":   (20,  80,  20),
    "rough":   (160, 200, 80),
}


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_features(
    bbox_wgs84: list,
    output_dir: Path,
    zoom: int = 17,
    boundary_data: Optional[dict] = None,
) -> dict:
    """
    Download satellite tiles and detect golf features using HSV segmentation.

    Args:
        bbox_wgs84:    [min_lon, min_lat, max_lon, max_lat]
        output_dir:    Directory for GeoJSON + debug images
        zoom:          Tile zoom level (17 ≈ 1.2m/px)
        boundary_data: Optional boundary dict (used for course-polygon masking)

    Returns:
        Vision summary dict compatible with run_pipeline.py.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  Vision: bbox={bbox_wgs84}, zoom={zoom}")

    # Download mosaic
    try:
        img_arr, transform_params = _download_satellite_mosaic(bbox_wgs84, zoom, output_dir)
    except Exception as e:
        log.warning(f"  Satellite tile download failed: {e}")
        return {"error": str(e), "detections": {}}

    if img_arr is None:
        return {"detections": {}}

    h, w = img_arr.shape[:2]
    log.info(f"  Satellite mosaic: {w}×{h} px")

    # Save mosaic
    try:
        _save_image(img_arr, output_dir / "satellite_mosaic.jpg")
    except Exception:
        pass

    # Optional course-boundary mask
    course_mask    = None
    course_polygon = None
    if boundary_data:
        try:
            from pipeline.course_mask import load_course_polygon, build_raster_mask
            course_polygon = load_course_polygon(boundary_data)
            course_mask    = build_raster_mask(
                course_polygon, transform_params, img_arr.shape
            )
        except Exception as e:
            log.debug(f"  Course mask setup failed (continuing without): {e}")

    # Run detectors
    detections  = {}
    all_masks   = {}
    filter_stats = {}
    detection_fns = [
        ("fairway", _detect_fairways),
        ("green",   _detect_greens),
        ("bunker",  _detect_bunkers),
        ("water",   _detect_water),
        ("trees",   _detect_trees),
        ("rough",   _detect_rough),
    ]

    for feat_type, fn in detection_fns:
        try:
            contours, mask = fn(img_arr)
            raw_count = len(contours)

            if course_mask is not None:
                try:
                    from pipeline.course_mask import apply_mask_to_contours
                    before    = len(contours)
                    contours  = apply_mask_to_contours(contours, course_mask)
                    if len(contours) < before:
                        log.debug(
                            f"  [{feat_type}] boundary mask: "
                            f"{before} → {len(contours)}"
                        )
                except Exception as me:
                    log.debug(f"  Mask apply failed for {feat_type}: {me}")

            after_boundary = len(contours)
            all_masks[feat_type] = mask

            out_path = output_dir / f"vision_{feat_type}s.geojson"
            _save_detections_geojson(
                contours, transform_params, feat_type, out_path,
                course_polygon=course_polygon,
            )
            try:
                saved_count = len(
                    json.loads(out_path.read_text(encoding="utf-8"))
                    .get("features", [])
                )
            except Exception:
                saved_count = len(contours)

            detections[feat_type] = {"count": saved_count, "path": str(out_path)}
            filter_stats[feat_type] = {
                "raw_contours":        raw_count,
                "after_boundary_mask": after_boundary,
                "after_area_clip":     saved_count,
            }
            log.info(
                f"  [{feat_type}] {raw_count} raw → "
                f"{after_boundary} in-bounds → {saved_count} after m² filter"
            )
        except Exception as e:
            log.warning(f"  Vision [{feat_type}] failed: {e}")

    # Write debug artifacts
    try:
        (output_dir / "class_filter_debug.json").write_text(
            json.dumps({"feature_counts": filter_stats}, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # Step 9 — vision debug overlay
    try:
        _save_vision_debug(img_arr, all_masks, output_dir / "vision_debug.png")
    except Exception as e:
        log.debug(f"  vision_debug.png failed: {e}")

    # Legacy overlay (for backward compat with existing code)
    try:
        _save_detection_overlay(img_arr, all_masks, output_dir / "vision_overlay.jpg")
    except Exception:
        pass

    summary = {
        "zoom":             zoom,
        "image_size":       [w, h],
        "transform_params": list(transform_params),
        "detections":       detections,
    }
    (output_dir / "vision_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def refilter_water_strict(output_dir: Path) -> int:
    """
    Post-hoc strict water filter.  Called when water count > 25.

    Applies minimum 2000 m² area + compactness ≥ 0.06 thresholds.
    Overwrites vision_waters.geojson in place.
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
        MIN_M2   = 2000
        MIN_COMP = 0.06
        filtered = []
        for f in features:
            gd = f.get("geometry")
            if not gd:
                continue
            try:
                poly     = shape(gd)
                poly_itm = shp_transform(t_to_itm.transform, poly)
                if poly_itm.area < MIN_M2:
                    continue
                comp = 4 * np.pi * poly_itm.area / max(poly_itm.length ** 2, 1e-9)
                if comp < MIN_COMP:
                    continue
                filtered.append(f)
            except Exception:
                continue
        _write_geojson_features(filtered, path)
        log.info(
            f"  Water strict refilter: {len(features)} → {len(filtered)} "
            f"(≥{MIN_M2}m², compactness ≥{MIN_COMP})"
        )
        return len(filtered)
    except Exception as e:
        log.warning(f"  refilter_water_strict failed: {e}")
        return 0


def merge_vision_with_osm(
    vision_summary: dict,
    osm_output_dir: Path,
    min_iou_threshold: float = 0.25,
) -> dict:
    """
    Merge vision detections with existing OSM GeoJSON features.

    Vision features with IoU < min_iou_threshold against OSM are appended.
    Overlapping features are unioned into the OSM geometry.
    """
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union

    stats: dict = {}
    feat_to_file = {
        "fairway": "fairways.geojson",
        "green":   "greens.geojson",
        "bunker":  "bunkers.geojson",
        "water":   "water.geojson",
        "trees":   "trees.geojson",
        "rough":   "rough.geojson",
    }

    for feat_type, det_info in vision_summary.get("detections", {}).items():
        vision_path = Path(det_info.get("path", ""))
        osm_path    = Path(osm_output_dir) / feat_to_file.get(feat_type,
                                                               f"{feat_type}s.geojson")
        if not vision_path.exists():
            continue

        vision_feats = _load_geojson(vision_path)
        osm_feats    = _load_geojson(osm_path) if osm_path.exists() else []

        osm_geoms = []
        for f in osm_feats:
            if f.get("geometry"):
                try:
                    osm_geoms.append(shape(f["geometry"]))
                except Exception:
                    pass
        osm_union = unary_union(osm_geoms) if osm_geoms else None

        new_feats:    List[dict] = []
        merged_count: int        = 0

        for vf in vision_feats:
            if not vf.get("geometry"):
                continue
            try:
                vg = shape(vf["geometry"])
            except Exception:
                continue

            if osm_union is None or not osm_union.intersects(vg):
                vf["properties"]["source"] = "vision"
                new_feats.append(vf)
                continue

            try:
                inter = osm_union.intersection(vg).area
                union = osm_union.area + vg.area - inter
                iou   = inter / max(union, 1e-10)
            except Exception:
                iou = 0.0

            if iou >= min_iou_threshold:
                continue   # already well covered by OSM

            # Try to union with the closest OSM feature
            merged = False
            for idx, of in enumerate(osm_feats):
                if not of.get("geometry"):
                    continue
                try:
                    og = shape(of["geometry"])
                    if og.intersects(vg):
                        u = og.union(vg)
                        if u.is_valid and not u.is_empty:
                            osm_feats[idx]["geometry"]             = mapping(u)
                            osm_feats[idx]["properties"]["source"] = "osm+vision"
                            merged = True
                            merged_count += 1
                            break
                except Exception:
                    pass
            if not merged:
                vf["properties"]["source"] = "vision"
                new_feats.append(vf)

        all_feats = osm_feats + new_feats
        _write_geojson_features(all_feats, osm_path)
        log.info(
            f"  Fusion [{feat_type}]: +{len(new_feats)} new, "
            f"{merged_count} merged with OSM → {osm_path.name}"
        )
        stats[feat_type] = len(new_feats)

    return stats


# ─── Satellite tile download ───────────────────────────────────────────────────

def _download_satellite_mosaic(
    bbox_wgs84: list,
    zoom: int,
    output_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Download and stitch satellite tiles into an RGB numpy array.

    Tries TILE_SOURCES (Google → ESRI → Bing) in priority order.
    Returns (rgb_array, transform_params) where:
        transform_params = (west_lon, north_lat, lon_per_px, lat_per_px)
    """
    try:
        import mercantile
        from PIL import Image
        import io as _io
        import requests as _req
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}")

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))

    if len(tiles) > 64 and zoom > 13:
        log.info(f"  Too many tiles ({len(tiles)}) at z{zoom} — stepping down")
        return _download_satellite_mosaic(bbox_wgs84, zoom - 1, output_dir)

    if not tiles:
        return None, None

    mosaic_size = getattr(config, "SATELLITE_MOSAIC_SIZE", 2048)
    log.info(f"  Fetching {len(tiles)} tiles at zoom {zoom}")

    session = _req.Session()
    tile_images: dict = {}

    for source in TILE_SOURCES:
        session.headers.update(source.get("headers", {}))
        fetched = _fetch_tiles(source, tiles, session)
        n       = len(fetched)
        log.info(f"  Tile source {source['name']}: {n}/{len(tiles)} tiles fetched")
        if n >= max(1, len(tiles) // 2):
            tile_images = fetched
            break

    if not tile_images:
        return None, None

    # Stitch
    all_tile_objs = [t for _, t in tile_images.values()]
    xs = [t.x for t in all_tile_objs]
    ys = [t.y for t in all_tile_objs]
    min_tx, max_tx = min(xs), max(xs)
    min_ty, max_ty = min(ys), max(ys)

    tw, th = 256, 256
    grid_w = (max_tx - min_tx + 1) * tw
    grid_h = (max_ty - min_ty + 1) * th

    mosaic = Image.new("RGB", (grid_w, grid_h), color=(80, 80, 80))
    for (tx, ty), (img, tile) in tile_images.items():
        mosaic.paste(img, ((tx - min_tx) * tw, (ty - min_ty) * th))

    # Resize to target mosaic size
    if max(grid_w, grid_h) != mosaic_size:
        scale = mosaic_size / max(grid_w, grid_h)
        new_w, new_h = int(grid_w * scale), int(grid_h * scale)
        mosaic = mosaic.resize((new_w, new_h), Image.LANCZOS)

    # Geographic transform
    ul = mercantile.bounds(mercantile.Tile(min_tx, min_ty, zoom))
    lr = mercantile.bounds(mercantile.Tile(max_tx, max_ty, zoom))
    out_w, out_h    = mosaic.size
    lon_per_px = (lr.east  - ul.west)  / out_w
    lat_per_px = (ul.north - lr.south) / out_h
    transform_params = (ul.west, ul.north, lon_per_px, lat_per_px)

    return np.array(mosaic, dtype=np.uint8), transform_params


def _fetch_tiles(source: dict, tiles: list, session) -> dict:
    """Fetch all tiles from one source. Returns {(tx,ty): (PIL.Image, tile)}."""
    from PIL import Image
    import io as _io

    url_template = source["url"]
    src_type     = source["type"]
    results      = {}

    for tile in tiles:
        try:
            if src_type == "quadkey":
                url = url_template.format(quadkey=_tile_to_quadkey(tile.x, tile.y, tile.z))
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
            log.debug(f"  Tile {tile} ({source['name']}): {e}")

    return results


def _tile_to_quadkey(x: int, y: int, z: int) -> str:
    qk = []
    for i in range(z, 0, -1):
        d = 0
        m = 1 << (i - 1)
        if x & m: d += 1
        if y & m: d += 2
        qk.append(str(d))
    return "".join(qk)


# ─── CV2 helper ───────────────────────────────────────────────────────────────

def _get_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise RuntimeError(
            "opencv-python required. Install: pip install opencv-python"
        )


# ─── Mask helpers ─────────────────────────────────────────────────────────────

def _hsv_mask(
    img_rgb: np.ndarray,
    h_range: tuple,
    s_range: tuple,
    v_range: tuple,
) -> np.ndarray:
    """Single-range HSV mask (OpenCV H 0-179)."""
    cv2 = _get_cv2()
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lo  = np.array([h_range[0], s_range[0], v_range[0]], dtype=np.uint8)
    hi  = np.array([h_range[1], s_range[1], v_range[1]], dtype=np.uint8)
    return cv2.inRange(hsv, lo, hi)


def _local_texture(img_rgb: np.ndarray, kernel: int = 7) -> np.ndarray:
    """Local texture = std-dev of the V channel (low = uniform surface)."""
    cv2 = _get_cv2()
    hsv   = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    v     = hsv[:, :, 2].astype(np.float32)
    vblur = cv2.blur(v, (kernel, kernel))
    vsq   = cv2.blur(v ** 2, (kernel, kernel))
    return np.sqrt(np.maximum(vsq - vblur ** 2, 0))


def _morph_clean(
    mask: np.ndarray,
    open_k: int = 5,
    close_k: int = 7,
    iterations: int = 1,
) -> np.ndarray:
    """OPEN (noise removal) → CLOSE (fill holes)."""
    cv2 = _get_cv2()
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=iterations)
    return mask


# ─── Feature detectors ────────────────────────────────────────────────────────

def _detect_fairways(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    cv2     = _get_cv2()
    t       = THRESHOLDS["fairway"]
    mask    = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture = _local_texture(img_rgb)
    mask[texture > t["texture_max"]] = 0
    mask    = _morph_clean(mask, open_k=7, close_k=9)
    # Extra pass to merge fragmented dogleg segments
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kc, iterations=3)

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
    cv2     = _get_cv2()
    t       = THRESHOLDS["green"]
    mask    = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture = _local_texture(img_rgb)
    mask[texture > t["texture_max"]] = 0
    mask    = _morph_clean(mask, open_k=3, close_k=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_GREEN_AREA_PX <= area <= MAX_GREEN_AREA_PX):
            continue
        peri = cv2.arcLength(c, True)
        if peri < 1:
            continue
        if 4 * np.pi * area / (peri ** 2) < MIN_GREEN_CIRC:
            continue
        valid.append(c)
    return valid, mask


def _detect_bunkers(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    cv2  = _get_cv2()
    mask = _hsv_mask(img_rgb, (14, 45), (10, 120), (155, 255))
    mask = _morph_clean(mask, open_k=3, close_k=5)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_BUNKER_AREA_PX <= area <= MAX_BUNKER_AREA_PX):
            continue
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) / max(min(w, h), 1) > MAX_BUNKER_ASPECT:
            continue
        valid.append(c)
    return valid, mask


def _detect_water(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    """
    Multi-signal water detection.

    Requires ≥ 3 of 4 signals to agree:
      1. NDWI ≈ (G-R)/(G+R) > 0.05
      2. Blue dominance
      3. HSV blue/cyan hue (H 95-135, dark V)
      4. Very dark V < 40 (excluding green/yellow hues)
    """
    cv2 = _get_cv2()

    r = img_rgb[:, :, 0].astype(np.float32)
    g = img_rgb[:, :, 1].astype(np.float32)
    b = img_rgb[:, :, 2].astype(np.float32)

    sig1 = ((g - r) / (g + r + 1e-6) > 0.05).astype(np.uint8)
    sig2 = ((b > r * 1.15) & (b > g * 0.85) & (b > 60)).astype(np.uint8)
    sig3 = (_hsv_mask(img_rgb, (95, 135), (40, 230), (0, 130)) > 0).astype(np.uint8)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    dark_v = (hsv[:, :, 2] < 40).astype(np.uint8)
    dark_v[(hsv[:, :, 0] >= 20) & (hsv[:, :, 0] <= 95)] = 0
    sig4 = dark_v

    mask = ((sig1 + sig2 + sig3 + sig4) >= 3).astype(np.uint8) * 255
    mask = _morph_clean(mask, open_k=7, close_k=11)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw = len(contours)

    MIN_PX   = 1000
    MAX_ASP  = 8.0
    MIN_COMP = 0.06
    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_PX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) / max(min(w, h), 1) > MAX_ASP:
            continue
        peri = cv2.arcLength(c, True)
        if peri > 0 and 4 * np.pi * area / (peri ** 2) < MIN_COMP:
            continue
        valid.append(c)

    log.info(
        f"  Water: {raw} raw → {len(valid)} after shape filter "
        f"(≥{MIN_PX}px, aspect ≤{MAX_ASP}, comp ≥{MIN_COMP})"
    )
    return valid, mask


def _detect_trees(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    cv2     = _get_cv2()
    t       = THRESHOLDS["trees"]
    mask    = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture = _local_texture(img_rgb)
    mask[texture < t["texture_min"]] = 0
    mask    = _morph_clean(mask, open_k=5, close_k=9)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if _get_cv2().contourArea(c) >= t["min_area_px"]], mask


def _detect_rough(img_rgb: np.ndarray) -> Tuple[list, np.ndarray]:
    cv2        = _get_cv2()
    t          = THRESHOLDS["rough"]
    tlo, thi   = t["texture_range"]
    mask       = _hsv_mask(img_rgb, t["h"], t["s"], t["v"])
    texture    = _local_texture(img_rgb)
    tex_mask   = ((texture >= tlo) & (texture <= thi)).astype(np.uint8) * 255
    mask       = cv2.bitwise_and(mask, tex_mask)
    mask       = _morph_clean(mask, open_k=7, close_k=11)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= t["min_area_px"]], mask


# ─── GeoJSON conversion ───────────────────────────────────────────────────────

def _save_detections_geojson(
    contours: list,
    transform_params: Optional[tuple],
    feature_type: str,
    out_path: Path,
    course_polygon=None,
) -> None:
    """
    Convert pixel contours → WGS84 GeoJSON polygons.

    Applies real-world m² area filters from config, optionally clips to
    the course boundary polygon, then saves.
    """
    from shapely.geometry import Polygon, mapping
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    if transform_params is None:
        _write_geojson_features([], out_path)
        return

    west, north, lon_per_px, lat_per_px = transform_params
    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)

    area_limits = {
        "bunker":  (getattr(config, "MIN_BUNKER_AREA_M2",  40),
                    getattr(config, "MAX_BUNKER_AREA_M2", 600)),
        "green":   (getattr(config, "MIN_GREEN_AREA_M2",  200),
                    getattr(config, "MAX_GREEN_AREA_M2", 2500)),
        "fairway": (getattr(config, "MIN_FAIRWAY_AREA_M2", 1000), None),
        "water":   (getattr(config, "MIN_WATER_AREA_M2",   50),
                    getattr(config, "MAX_WATER_AREA_M2", 500000)),
    }
    min_m2, max_m2 = area_limits.get(feature_type, (0, None))

    features = []
    for c in contours:
        pts  = c.reshape(-1, 2)
        coords = [
            (west + x * lon_per_px, north - y * lat_per_px)
            for x, y in pts
        ]
        if len(coords) < 3:
            continue
        coords.append(coords[0])

        try:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue

            # Real-world area filter
            try:
                poly_itm = shp_transform(t_to_itm.transform, poly)
                area_m2  = poly_itm.area
                if area_m2 < min_m2:
                    continue
                if max_m2 is not None and area_m2 > max_m2:
                    continue
            except Exception:
                pass

            # Course boundary clip
            if course_polygon is not None:
                try:
                    clipped = poly.intersection(course_polygon)
                    if clipped.is_empty:
                        continue
                    poly = clipped
                except Exception:
                    pass

            features.append({
                "type": "Feature",
                "geometry":   mapping(poly),
                "properties": {
                    "type":   feature_type,
                    "source": "satellite_vision",
                },
            })
        except Exception:
            continue

    _write_geojson_features(features, out_path)


# ─── Debug visualisation (Step 9) ─────────────────────────────────────────────

def _save_vision_debug(
    img_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """
    Step 9 — vision_debug.png

    Draws semi-transparent coloured overlays for each detected feature type
    on top of the satellite mosaic.
    """
    cv2 = _get_cv2()
    canvas  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    overlay = canvas.copy()

    layer_order = ["water", "bunker", "fairway", "green", "trees", "rough"]
    for feat_type in layer_order:
        mask = masks.get(feat_type)
        if mask is None:
            continue
        rgb    = _DBG_RGB.get(feat_type, (200, 200, 200))
        colour = (rgb[2], rgb[1], rgb[0])   # BGR for OpenCV
        coloured = np.zeros_like(canvas)
        coloured[mask > 0] = colour
        cv2.addWeighted(coloured, 0.40, overlay, 1.0, 0, overlay)

    # Blend overlay onto canvas
    cv2.addWeighted(overlay, 0.60, canvas, 0.40, 0, canvas)

    # Legend
    font  = cv2.FONT_HERSHEY_SIMPLEX
    ly    = 24
    for feat_type in layer_order:
        rgb    = _DBG_RGB.get(feat_type, (200, 200, 200))
        colour = (rgb[2], rgb[1], rgb[0])
        cv2.rectangle(canvas, (8, ly - 12), (24, ly + 4), colour, -1)
        cv2.putText(canvas, feat_type, (28, ly + 2), font, 0.45,
                    (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, feat_type, (28, ly + 2), font, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        ly += 22

    cv2.imwrite(str(out_path), canvas)
    log.info(f"  vision_debug.png → {out_path.name}")


def _save_detection_overlay(
    img_rgb: np.ndarray,
    masks: Dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Legacy coloured overlay for backward compat (vision_overlay.jpg)."""
    _save_vision_debug(img_rgb, masks, out_path)


def _save_image(img_rgb: np.ndarray, out_path: Path, quality: int = 88) -> None:
    from PIL import Image
    Image.fromarray(img_rgb).save(str(out_path), quality=quality)


def _save_debug_image(img_rgb: np.ndarray, out_path: Path) -> None:
    _save_image(img_rgb, out_path)


# ─── GeoJSON utilities ────────────────────────────────────────────────────────

def _write_geojson_features(features: list, path: Path) -> None:
    path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )


def _load_geojson(path: Path) -> list:
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("features", [])
    except Exception:
        return []

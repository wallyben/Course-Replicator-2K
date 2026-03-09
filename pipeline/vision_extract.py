"""
vision_extract.py — Satellite imagery feature detection via computer vision.

UPGRADE 3: Detect golf features missing from OSM using satellite imagery.

Method:
  1. Download satellite image tiles (OpenStreetMap tile server)
  2. Convert to HSV colour space
  3. Segment fairways via green-band analysis
  4. Detect bunkers via sand colour clustering
  5. Detect greens via low-texture circular patch detection

Libraries: opencv-python, scikit-image, numpy
Output: Merges detections with OSM geometry (supplementary GeoJSON)
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)


# ─── Colour thresholds (HSV) ──────────────────────────────────────────────────

# Fairway green: medium-dark green turf
FAIRWAY_HSV = {
    "h_min": 35, "h_max": 90,    # green hue range
    "s_min": 40, "s_max": 255,   # saturated
    "v_min": 40, "v_max": 200,   # not too dark/bright
}

# Bunker sand: pale yellow/beige
BUNKER_HSV = {
    "h_min": 15, "h_max": 40,
    "s_min": 20, "s_max": 120,
    "v_min": 160, "v_max": 255,
}

# Green (putting green): similar to fairway but smaller and more uniform
GREEN_HSV = {
    "h_min": 40, "h_max": 80,
    "s_min": 50, "s_max": 200,
    "v_min": 60, "v_max": 180,
}

# Minimum pixel areas for detection (at zoom 17, ~1.2m/px)
MIN_FAIRWAY_PX  = 500
MIN_BUNKER_PX   = 30
MIN_GREEN_PX    = 100
MAX_GREEN_PX    = 800   # greens are small


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_features(
    bbox_wgs84: list,
    output_dir: Path,
    zoom: int = 17,
) -> dict:
    """
    Download satellite tiles and detect golf features using computer vision.

    Args:
        bbox_wgs84:  [min_lon, min_lat, max_lon, max_lat]
        output_dir:  Directory for output GeoJSON and debug images
        zoom:        Tile zoom level (17 ≈ 1.2m/px, 16 ≈ 2.4m/px)

    Returns:
        Detection summary dict with detected feature GeoJSON paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Vision extraction: bbox={bbox_wgs84}, zoom={zoom}")

    # Step 1: Download and stitch satellite tiles
    try:
        img_arr, img_transform, img_crs = _download_satellite_mosaic(
            bbox_wgs84, zoom, output_dir
        )
    except Exception as e:
        log.warning(f"Satellite tile download failed: {e}. Skipping vision extraction.")
        return {"error": str(e), "detections": {}}

    if img_arr is None:
        log.warning("No satellite tiles available — skipping vision extraction")
        return {"detections": {}}

    log.info(f"Satellite mosaic: {img_arr.shape[1]}×{img_arr.shape[0]}px")

    # Step 2: Run CV detections
    detections = {}

    try:
        fairway_polys = _detect_fairways(img_arr)
        fairway_path = output_dir / "vision_fairways.geojson"
        _save_detections_geojson(
            fairway_polys, img_transform, "fairway", fairway_path
        )
        detections["fairway"] = {"count": len(fairway_polys), "path": str(fairway_path)}
        log.info(f"Vision: {len(fairway_polys)} fairway regions detected")
    except Exception as e:
        log.warning(f"Fairway detection failed: {e}")

    try:
        bunker_polys = _detect_bunkers(img_arr)
        bunker_path = output_dir / "vision_bunkers.geojson"
        _save_detections_geojson(
            bunker_polys, img_transform, "bunker", bunker_path
        )
        detections["bunker"] = {"count": len(bunker_polys), "path": str(bunker_path)}
        log.info(f"Vision: {len(bunker_polys)} bunker regions detected")
    except Exception as e:
        log.warning(f"Bunker detection failed: {e}")

    try:
        green_polys = _detect_greens(img_arr)
        green_path = output_dir / "vision_greens.geojson"
        _save_detections_geojson(
            green_polys, img_transform, "green", green_path
        )
        detections["green"] = {"count": len(green_polys), "path": str(green_path)}
        log.info(f"Vision: {len(green_polys)} putting green candidates detected")
    except Exception as e:
        log.warning(f"Green detection failed: {e}")

    # Save debug composite
    try:
        _save_debug_image(img_arr, output_dir / "vision_debug.jpg")
    except Exception:
        pass

    summary = {
        "zoom":       zoom,
        "image_size": [img_arr.shape[1], img_arr.shape[0]],
        "detections": detections,
    }
    (output_dir / "vision_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def merge_vision_with_osm(
    vision_summary: dict,
    osm_output_dir: Path,
    min_iou_threshold: float = 0.3,
) -> dict:
    """
    Merge vision detections with OSM features.
    Adds vision-detected features that have low IoU overlap with existing OSM features.

    Args:
        vision_summary:   Output of detect_features()
        osm_output_dir:   Directory containing OSM GeoJSON files
        min_iou_threshold: Features with IoU above this threshold are duplicates

    Returns:
        Merge statistics dict.
    """
    from shapely.geometry import shape
    from shapely.ops import unary_union

    stats = {}
    for feat_type, det_info in vision_summary.get("detections", {}).items():
        vision_path = Path(det_info.get("path", ""))
        osm_path = osm_output_dir / f"{feat_type}s.geojson"

        if not vision_path.exists():
            continue

        vision_feats = _load_geojson(vision_path)
        osm_feats = _load_geojson(osm_path) if osm_path.exists() else []

        osm_geoms = [shape(f["geometry"]) for f in osm_feats if f.get("geometry")]
        osm_union = unary_union(osm_geoms) if osm_geoms else None

        added = []
        for vf in vision_feats:
            if not vf.get("geometry"):
                continue
            vg = shape(vf["geometry"])
            if osm_union is None or not osm_union.intersects(vg):
                vf["properties"]["source"] = "vision"
                added.append(vf)
            else:
                inter = osm_union.intersection(vg)
                iou = inter.area / (osm_union.area + vg.area - inter.area)
                if iou < min_iou_threshold:
                    vf["properties"]["source"] = "vision"
                    added.append(vf)

        if added and osm_path.exists():
            existing = _load_geojson(osm_path)
            merged = existing + added
            osm_path.write_text(json.dumps(
                {"type": "FeatureCollection", "features": merged}, indent=2
            ))
            log.info(f"Vision merge: added {len(added)} {feat_type}(s) to {osm_path.name}")
        elif added:
            osm_path.write_text(json.dumps(
                {"type": "FeatureCollection", "features": added}, indent=2
            ))

        stats[feat_type] = len(added)

    return stats


# ─── Satellite tile download ──────────────────────────────────────────────────

def _download_satellite_mosaic(
    bbox_wgs84: list,
    zoom: int,
    output_dir: Path,
) -> Tuple[Optional[np.ndarray], object, str]:
    """
    Download and stitch satellite tiles into a single mosaic.
    Uses OpenStreetMap tile server (imagery from ESRI/Mapbox).

    Returns:
        (rgb_array [H,W,3], affine_transform, crs_epsg_str)
        or (None, None, None) on failure
    """
    try:
        import mercantile
        from PIL import Image
        import io as _io
        import requests
    except ImportError as e:
        raise RuntimeError(f"Missing dependency: {e}. Install mercantile, Pillow")

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84

    # Use ESRI World Imagery tiles (free, no key required for reasonable usage)
    TILE_URLS = [
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "https://tile.openstreetmap.org/{z}/{x}/{y}.png",   # fallback (no satellite)
    ]

    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zooms=zoom))
    if not tiles:
        return None, None, None
    if len(tiles) > 64:
        log.warning(f"Too many tiles ({len(tiles)}) at zoom {zoom}, clamping to zoom {zoom-1}")
        return _download_satellite_mosaic(bbox_wgs84, zoom - 1, output_dir)

    log.info(f"Downloading {len(tiles)} satellite tiles at zoom {zoom}")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "CourseReplicator2K/2.0 (geospatial research)"
    })

    tile_images = {}   # (x, y) → PIL Image
    tile_url_template = TILE_URLS[0]

    for tile in tiles:
        url = tile_url_template.format(z=tile.z, x=tile.x, y=tile.y)
        try:
            resp = session.get(url, timeout=15)
            if resp.status_code == 200:
                img = Image.open(_io.BytesIO(resp.content)).convert("RGB")
                tile_images[(tile.x, tile.y)] = (img, tile)
            else:
                log.debug(f"Tile {tile} HTTP {resp.status_code}")
        except Exception as e:
            log.debug(f"Tile {tile} failed: {e}")

    if not tile_images:
        return None, None, None

    # Determine grid extent
    all_tiles = [t for _, (_, t) in tile_images.items()]
    x_coords = [t.x for t in all_tiles]
    y_coords = [t.y for t in all_tiles]
    min_tx, max_tx = min(x_coords), max(x_coords)
    min_ty, max_ty = min(y_coords), max(y_coords)

    tile_w = 256
    tile_h = 256
    grid_w = (max_tx - min_tx + 1) * tile_w
    grid_h = (max_ty - min_ty + 1) * tile_h

    mosaic = Image.new("RGB", (grid_w, grid_h), color=(128, 128, 128))

    for (tx, ty), (img, tile) in tile_images.items():
        px = (tx - min_tx) * tile_w
        py = (ty - min_ty) * tile_h
        mosaic.paste(img, (px, py))

    # Compute affine transform
    ul_tile = mercantile.Tile(min_tx, min_ty, zoom)
    ul_bounds = mercantile.bounds(ul_tile)
    lr_tile = mercantile.Tile(max_tx, max_ty, zoom)
    lr_bounds = mercantile.bounds(lr_tile)

    total_west  = ul_bounds.west
    total_north = ul_bounds.north
    total_east  = lr_bounds.east
    total_south = lr_bounds.south

    lon_per_px = (total_east - total_west) / grid_w
    lat_per_px = (total_north - total_south) / grid_h

    # Simple affine: [lon_per_px, 0, west, 0, -lat_per_px, north]
    # Store as tuple for use in _pixel_to_wgs84
    transform_params = (total_west, total_north, lon_per_px, lat_per_px)

    arr = np.array(mosaic, dtype=np.uint8)
    return arr, transform_params, "EPSG:4326"


# ─── Feature detection algorithms ────────────────────────────────────────────

def _detect_fairways(img_rgb: np.ndarray) -> List[dict]:
    """
    Detect fairway regions using HSV green segmentation.
    Returns list of pixel contour dicts.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required for vision extraction")

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower = np.array([FAIRWAY_HSV["h_min"], FAIRWAY_HSV["s_min"], FAIRWAY_HSV["v_min"]])
    upper = np.array([FAIRWAY_HSV["h_max"], FAIRWAY_HSV["s_max"], FAIRWAY_HSV["v_max"]])
    mask = cv2.inRange(hsv, lower, upper)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_FAIRWAY_PX]


def _detect_bunkers(img_rgb: np.ndarray) -> List[dict]:
    """
    Detect bunker regions using sand-colour HSV segmentation.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required for vision extraction")

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower = np.array([BUNKER_HSV["h_min"], BUNKER_HSV["s_min"], BUNKER_HSV["v_min"]])
    upper = np.array([BUNKER_HSV["h_max"], BUNKER_HSV["s_max"], BUNKER_HSV["v_max"]])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in contours if cv2.contourArea(c) >= MIN_BUNKER_PX]


def _detect_greens(img_rgb: np.ndarray) -> List[dict]:
    """
    Detect putting greens as small, compact, low-texture green patches.
    Uses: HSV green mask + circularity filter + texture analysis.
    """
    try:
        import cv2
    except ImportError:
        raise RuntimeError("opencv-python required for vision extraction")

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    lower = np.array([GREEN_HSV["h_min"], GREEN_HSV["s_min"], GREEN_HSV["v_min"]])
    upper = np.array([GREEN_HSV["h_max"], GREEN_HSV["s_max"], GREEN_HSV["v_max"]])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    green_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_GREEN_PX or area > MAX_GREEN_PX:
            continue

        # Circularity: 4π·area/perimeter²  (1.0 = perfect circle)
        perimeter = cv2.arcLength(c, True)
        if perimeter < 1:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.4:   # reject very irregular shapes
            continue

        # Low texture: stddev of V channel within contour ROI
        x, y, w, h = cv2.boundingRect(c)
        roi_v = hsv[y:y+h, x:x+w, 2]
        texture = float(np.std(roi_v))
        if texture > 40:   # high texture = rough, not a putting green
            continue

        green_candidates.append(c)

    return green_candidates


# ─── GeoJSON conversion ───────────────────────────────────────────────────────

def _save_detections_geojson(
    contours: list,
    transform_params: tuple,
    feature_type: str,
    out_path: Path,
) -> None:
    """
    Convert pixel contours to WGS84 GeoJSON and save.
    transform_params: (west, north, lon_per_px, lat_per_px)
    """
    from shapely.geometry import Polygon, mapping
    import cv2

    if transform_params is None:
        return

    west, north, lon_per_px, lat_per_px = transform_params
    features = []

    for contour in contours:
        if len(contour) < 3:
            continue
        pts = contour.squeeze()
        if pts.ndim != 2 or pts.shape[1] != 2:
            continue

        # Pixel → WGS84
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
            if poly.area < 1e-10:
                continue
            features.append({
                "type": "Feature",
                "geometry": mapping(poly),
                "properties": {
                    "type":   feature_type,
                    "source": "vision",
                    "area_px": float(cv2.contourArea(contour)),
                },
            })
        except Exception:
            continue

    geojson = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(geojson, indent=2))


def _save_debug_image(img_rgb: np.ndarray, out_path: Path) -> None:
    """Save the satellite mosaic as a JPEG for visual debugging."""
    from PIL import Image
    img = Image.fromarray(img_rgb)
    img.save(str(out_path), quality=80)


def _load_geojson(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("features", [])
    except Exception:
        return []

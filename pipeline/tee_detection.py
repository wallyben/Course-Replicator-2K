"""
tee_detection.py — Satellite-based tee box detection.

Tees are rarely present in OSM, so this module detects them from
satellite imagery using geometric and colour heuristics.

Tee box characteristics:
  - Rectangular or oval shape
  - Very short, uniform grass (similar colour to fairways)
  - Located 5–80m from the fairway start point
  - Area typically 50–500 m²

Algorithm:
  1. Load fairway polygons from OSM GeoJSON
  2. For each fairway, find its start-end axis and extract candidate tee region
  3. Detect small rectangular turf patches in the satellite mosaic near the start
  4. Verify colour similarity to fairway and geometry constraints
  5. Output tees.geojson

If the satellite mosaic is unavailable, falls back to geometric estimation
from OSM fairway start points only.

Output:
  tees.geojson — GeoJSON FeatureCollection with per-tee properties:
    tee_centroid  [lon, lat]
    tee_area      float (m²)
    tee_orientation  float (degrees)
    confidence    float (0.0–1.0)
    source        "vision" | "estimated"
"""

import json
import logging
import math
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

# Tee size constraints (m²)
MIN_TEE_AREA_M2 = 30
MAX_TEE_AREA_M2 = 500

# Distance from fairway start to search for tee (m)
TEE_SEARCH_MIN_M = getattr(config, "TEE_SEARCH_MIN_M", 5)
TEE_SEARCH_MAX_M = getattr(config, "TEE_SEARCH_MAX_M", 80)

# Minimum colour similarity (HSV hue distance) to fairway
MAX_HUE_DIFF = 20   # degrees in OpenCV H space (0-179)


# ─── Public API ───────────────────────────────────────────────────────────────

def detect_tees(
    bbox_wgs84: list,
    output_dir: Path,
    satellite_mosaic_path: Optional[Path] = None,
) -> dict:
    """
    Detect tee boxes and write tees.geojson.

    Args:
        bbox_wgs84:             [min_lon, min_lat, max_lon, max_lat]
        output_dir:             Directory to write tees.geojson
        satellite_mosaic_path:  Path to satellite_mosaic.jpg (if available)

    Returns:
        Summary dict with tee count and output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tees_path  = output_dir / "tees.geojson"

    # Load fairways from OSM GeoJSON (needed in both paths)
    fairway_geoms = _load_fairway_geoms(output_dir)

    tees = []

    # Path A: vision-based detection (if mosaic available)
    if satellite_mosaic_path and Path(satellite_mosaic_path).exists():
        try:
            tees = _detect_tees_vision(
                bbox_wgs84, fairway_geoms, Path(satellite_mosaic_path)
            )
            log.info(f"Tee detection (vision): {len(tees)} tees found")
        except Exception as e:
            log.warning(f"Vision tee detection failed: {e} — falling back to geometric")
            tees = []

    # Path B: geometric estimation from fairway start points (fallback)
    if not tees and fairway_geoms:
        try:
            tees = _estimate_tees_geometric(fairway_geoms)
            log.info(f"Tee detection (geometric): {len(tees)} tees estimated")
        except Exception as e:
            log.warning(f"Geometric tee estimation failed: {e}")
            tees = []

    _write_tees_geojson(tees, tees_path)

    return {
        "tee_count":  len(tees),
        "tees_path":  str(tees_path),
        "tees":       tees,
    }


# ─── Vision-based detection ───────────────────────────────────────────────────

def _detect_tees_vision(
    bbox_wgs84: list,
    fairway_geoms: list,
    mosaic_path: Path,
) -> List[dict]:
    """
    Detect tee boxes in satellite imagery using:
      1. Fairway-colour candidate extraction
      2. Rectangular geometry filter
      3. Proximity to fairway start

    Returns list of tee dicts.
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        raise RuntimeError("opencv-python and Pillow required for vision tee detection")

    from shapely.geometry import shape, Point
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    img_pil = Image.open(str(mosaic_path)).convert("RGB")
    img     = np.array(img_pil, dtype=np.uint8)
    h_px, w_px = img.shape[:2]

    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    lon_per_px = (max_lon - min_lon) / w_px
    lat_per_px = (max_lat - min_lat) / h_px

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)

    def px_to_wgs(px, py):
        return (
            min_lon + px * lon_per_px,
            max_lat - py * lat_per_px,   # image Y is inverted
        )

    def wgs_to_px(lon, lat):
        return (
            int((lon - min_lon) / lon_per_px),
            int((max_lat - lat) / lat_per_px),
        )

    # Sample the fairway colour (median HSV in fairway mask area)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    fairway_mask = _build_fairway_mask(img, cv2)

    if cv2.countNonZero(fairway_mask) < 100:
        log.debug("Tee detection: insufficient fairway pixels for colour reference")
        return []

    fw_hsv   = hsv[fairway_mask > 0]
    fw_h_med = float(np.median(fw_hsv[:, 0]))
    fw_s_med = float(np.median(fw_hsv[:, 1]))

    # Find small compact green patches (tee candidates)
    # Tees have the same hue as fairways but are very small rectangular patches
    h_lo = max(0,   int(fw_h_med - MAX_HUE_DIFF))
    h_hi = min(179, int(fw_h_med + MAX_HUE_DIFF))
    lo   = np.array([h_lo, max(0, int(fw_s_med - 60)), 80], dtype=np.uint8)
    hi   = np.array([h_hi, min(255, int(fw_s_med + 60)), 220], dtype=np.uint8)
    candidate_mask = cv2.inRange(hsv, lo, hi)

    # Morphological cleanup — small kernel to preserve individual tee boxes
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN,  k, iterations=1)
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, k, iterations=2)

    # Subtract the main fairway mask — tees are NOT already part of fairways
    candidate_mask = cv2.bitwise_and(candidate_mask, cv2.bitwise_not(fairway_mask))

    contours, _ = cv2.findContours(
        candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    tees = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px < 30 or area_px > 2000:   # pixel range
            continue

        # Fit a rotated rectangle to assess rectangularity
        rect  = cv2.minAreaRect(c)
        box_w = max(rect[1][0], rect[1][1])
        box_h = min(rect[1][0], rect[1][1])
        if box_h < 1:
            continue
        aspect = box_w / box_h
        if aspect > 8:   # too thin — not a tee
            continue

        # Rectangle fill ratio (area / bounding box area)
        fill = area_px / max(box_w * box_h, 1)
        if fill < 0.4:  # too irregular for a tee
            continue

        # Centroid in WGS84
        M   = cv2.moments(c)
        if M["m00"] < 1:
            continue
        cx_px = int(M["m10"] / M["m00"])
        cy_px = int(M["m01"] / M["m00"])
        lon, lat = px_to_wgs(cx_px, cy_px)

        # Check proximity to a fairway start point
        tee_pt_itm = Point(*t_to_itm.transform(lon, lat))
        near_fairway = False
        for fw_geom in fairway_geoms:
            try:
                fw_itm   = shp_transform(t_to_itm.transform, fw_geom)
                fw_start = _fairway_start_point(fw_itm)
                dist     = tee_pt_itm.distance(fw_start)
                if TEE_SEARCH_MIN_M <= dist <= TEE_SEARCH_MAX_M:
                    near_fairway = True
                    break
            except Exception:
                continue

        if not near_fairway:
            continue

        # Estimate real-world area
        try:
            from shapely.geometry import Polygon
            pts_wgs = [px_to_wgs(pt[0][0], pt[0][1]) for pt in c]
            if len(pts_wgs) >= 3:
                poly_wgs = Polygon(pts_wgs)
                poly_itm = shp_transform(t_to_itm.transform, poly_wgs)
                area_m2  = poly_itm.area
            else:
                area_m2 = float(area_px) * (lon_per_px * 111000) ** 2
        except Exception:
            area_m2 = float(area_px) * (lon_per_px * 111000) ** 2

        if not (MIN_TEE_AREA_M2 <= area_m2 <= MAX_TEE_AREA_M2):
            continue

        orientation = float(rect[2])  # degrees

        tees.append({
            "tee_centroid":    [lon, lat],
            "tee_area":        round(area_m2, 1),
            "tee_orientation": round(orientation, 1),
            "confidence":      0.65,
            "source":          "vision",
        })

    return tees


def _build_fairway_mask(img_rgb: np.ndarray, cv2) -> np.ndarray:
    """Build a binary mask of the main fairway colour in the image."""
    hsv  = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lo   = np.array([35, 40, 50],  dtype=np.uint8)
    hi   = np.array([85, 220, 200], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    k    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
    return mask


def _fairway_start_point(fairway_itm):
    """
    Return the 'start' point of a fairway polygon in ITM coordinates.
    Heuristic: the point on the fairway boundary with the smallest Y (south)
    in the ITM bounding box is likely the tee end.
    """
    from shapely.geometry import Point
    bounds = fairway_itm.bounds   # (minx, miny, maxx, maxy)
    return Point(
        (bounds[0] + bounds[2]) / 2,
        bounds[1],
    )


# ─── Geometric fallback ───────────────────────────────────────────────────────

def _estimate_tees_geometric(fairway_geoms: list) -> List[dict]:
    """
    Estimate tee positions from fairway start points when satellite detection
    is unavailable.  Places a synthetic 8×12m tee box 30m from the fairway start.
    """
    from shapely.geometry import Point, Polygon, mapping
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    t_to_wgs = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

    tees = []
    for fw_geom in fairway_geoms:
        try:
            fw_itm   = shp_transform(t_to_itm.transform, fw_geom)
            fw_start = _fairway_start_point(fw_itm)

            # Tee box: 8×12m rectangle 30m south of fairway start
            tee_cx = fw_start.x
            tee_cy = fw_start.y - 30

            # Build 8×12m rectangle
            half_w, half_h = 4.0, 6.0
            box_itm = Polygon([
                (tee_cx - half_w, tee_cy - half_h),
                (tee_cx + half_w, tee_cy - half_h),
                (tee_cx + half_w, tee_cy + half_h),
                (tee_cx - half_w, tee_cy + half_h),
                (tee_cx - half_w, tee_cy - half_h),
            ])
            box_wgs   = shp_transform(t_to_wgs.transform, box_itm)
            centroid  = box_wgs.centroid

            tees.append({
                "tee_centroid":    [centroid.x, centroid.y],
                "tee_area":        round(box_itm.area, 1),
                "tee_orientation": 0.0,
                "confidence":      0.35,
                "source":          "estimated",
                "geometry":        mapping(box_wgs),
            })
        except Exception as e:
            log.debug(f"Geometric tee estimate failed for fairway: {e}")
            continue

    return tees


# ─── Utilities ────────────────────────────────────────────────────────────────

def _load_fairway_geoms(output_dir: Path) -> list:
    """Load fairway polygons from OSM GeoJSON output."""
    from shapely.geometry import shape

    fw_path = output_dir / "fairways.geojson"
    if not fw_path.exists():
        return []
    try:
        data = json.loads(fw_path.read_text(encoding="utf-8"))
        geoms = []
        for f in data.get("features", []):
            if f.get("geometry"):
                try:
                    geoms.append(shape(f["geometry"]))
                except Exception:
                    pass
        return geoms
    except Exception:
        return []


def _write_tees_geojson(tees: List[dict], out_path: Path) -> None:
    """Write tees as a GeoJSON FeatureCollection."""
    from shapely.geometry import Point, mapping

    features = []
    for tee in tees:
        lon, lat = tee["tee_centroid"]
        geom = tee.get("geometry") or mapping(Point(lon, lat))
        features.append({
            "type":     "Feature",
            "geometry": geom,
            "properties": {
                "tee_centroid":    tee["tee_centroid"],
                "tee_area":        tee.get("tee_area"),
                "tee_orientation": tee.get("tee_orientation"),
                "confidence":      tee.get("confidence", 0.5),
                "source":          tee.get("source", "unknown"),
            },
        })

    out_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    log.info(f"Tee detection: {len(features)} tees written to {out_path.name}")

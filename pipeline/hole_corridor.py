"""
hole_corridor.py — Corridor-first hole reconstruction engine.

Rationale
---------
Fairway HSV detection fails on parkland courses where fairways share
similar colour with rough, and tree shadows fragment the colour signal.
Old Conna example: only 2 fairways detected vs 18 expected.

Greens (circular, compact, distinctive) and tee boxes (rectangular,
elevated) are reliably detected.  This module reconstructs holes by
working backward from greens:

    green → find/synthesize tee → compute corridor → generate fairway

Algorithm
---------
1. Load 18 greens (primary anchor — reliably detected circular patches)
2. For each green, find the nearest unassigned tee cluster ≤550m away
3. Greens with no matched tee get a synthesized tee (80–180m away from
   the course centre — the expected approach direction for an 18-hole course)
4. For each (tee, green) pair:
   a. Define a search corridor: parallelogram ±55m either side of
      the tee→green axis, in the satellite mosaic pixel space
   b. Build a playable grass mask within that corridor:
        grass_mask = HSV (H:28-95, S:25-220, V:45-215)
        tree_mask  = dark (V<120) + high-texture green
        playable   = grass_mask AND NOT tree_mask
   c. Morphological closing (9px, 2×) to fill tree-gap fragments
   d. Largest connected component = corridor polygon
   e. Skeletonize → centreline (sorted along tee→green axis)
5. Validate each hole: 80m ≤ length ≤ 700m
6. Write corridor_fairways.geojson; overwrite fairways.geojson if sparse

Output holes are compatible with routing.py dict format.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ── Tunable constants ─────────────────────────────────────────────────────────
MIN_HOLE_M        = 80.0   # minimum valid hole length (m)
MAX_HOLE_M        = 700.0  # maximum valid hole length (m)
MAX_TEE_MATCH_M   = 550.0  # max tee→green distance to form a matched pair
CORRIDOR_HALF_W_M = 55.0   # search strip half-width each side of axis (m)
SYNTH_TEE_DIST_M  = 130.0  # distance for synthesized tee from green
MIN_GRASS_FRAC    = 0.18   # minimum playable fraction of corridor strip


# ── Public API ────────────────────────────────────────────────────────────────

def reconstruct_corridors(
    tees: List[dict],
    greens: List[dict],
    output_dir: Path,
) -> List[dict]:
    """
    Reconstruct holes using green-anchored corridor approach.

    Args:
        tees:       [{lon, lat, ...}] — tee cluster positions (may be few)
        greens:     [{lon, lat, ...}] — green centroids (primary anchors)
        output_dir: pipeline output directory (satellite_mosaic.jpg + boundary.json)

    Returns:
        List of hole dicts in routing.py format, or [] if reconstruction
        fails (caller falls back to legacy methods).
    """
    if not greens:
        log.warning("Corridor routing: no greens available — skipping")
        return []

    output_dir = Path(output_dir)

    # Load satellite mosaic + transform parameters
    mosaic_arr, transform_params = _load_mosaic(output_dir)
    if mosaic_arr is None or transform_params is None:
        log.warning("Corridor routing: satellite mosaic unavailable — skipping")
        return []

    log.info(
        f"Corridor routing: {len(tees)} tee clusters, {len(greens)} greens, "
        f"mosaic {mosaic_arr.shape[1]}×{mosaic_arr.shape[0]}px"
    )

    # Build playable masks once for the whole mosaic (reused per pair)
    grass_mask, tree_mask = _build_playable_masks(mosaic_arr)

    # Pair greens to tees (green-anchored — synthesize missing tees)
    course_center = _course_center(greens)
    pairs = _green_anchored_pairs(tees, greens, course_center)
    n_matched   = sum(1 for *_, s in pairs if s == "matched")
    n_synth     = sum(1 for *_, s in pairs if s == "synthesized")
    log.info(
        f"Corridor routing: {len(pairs)} pairs — "
        f"{n_matched} matched tees, {n_synth} synthesized tees"
    )

    holes = []
    for tee_pos, green_pos, source in pairs:
        try:
            dist_m = _haversine(
                tee_pos["lon"], tee_pos["lat"],
                green_pos["lon"], green_pos["lat"],
            )
            if not (MIN_HOLE_M <= dist_m <= MAX_HOLE_M):
                log.debug(
                    f"Corridor pair skipped: {dist_m:.0f}m out of "
                    f"[{MIN_HOLE_M:.0f},{MAX_HOLE_M:.0f}]"
                )
                continue

            # Compute corridor polygon + centreline from satellite
            corr = _compute_corridor(
                tee_pos, green_pos,
                mosaic_arr, transform_params,
                grass_mask, tree_mask,
            )

            conf = 0.75 if source == "matched" else 0.52
            hole = {
                "hole_number":         len(holes) + 1,
                "tee_position":        {"lon": tee_pos["lon"], "lat": tee_pos["lat"]},
                "green_position":      {"lon": green_pos["lon"], "lat": green_pos["lat"]},
                "distance_m":          round(dist_m, 1),
                "distance_yards":      round(dist_m * 1.09361, 0),
                "estimated_length_m":  round(dist_m, 1),
                "estimated_length_yd": round(dist_m * 1.09361, 0),
                "par":                 _estimate_par(dist_m),
                "routing_source":      f"corridor_{source}",
                "confidence":          conf,
                "source_mix":          "corridor+green_anchor",
                "tee_centroid":        [tee_pos["lon"], tee_pos["lat"]],
                "green_centroid":      [green_pos["lon"], green_pos["lat"]],
            }
            if corr:
                hole["fairway_corridor"]  = corr.get("polygon")
                hole["fairway_centerline"] = corr.get("centerline")

            holes.append(hole)

        except Exception as e:
            log.debug(f"Corridor pair failed: {e}")
            continue

    log.info(f"Corridor routing: {len(holes)} valid holes from {len(pairs)} pairs")

    if holes:
        _write_corridor_fairways(holes, output_dir)

    return holes


# ── Green-anchored pairing ─────────────────────────────────────────────────────

def _green_anchored_pairs(
    tees: List[dict],
    greens: List[dict],
    course_center: Tuple[float, float],
) -> List[Tuple[dict, dict, str]]:
    """
    For every green, find the nearest unassigned tee within MAX_TEE_MATCH_M.
    Greens with no match get a synthesized tee placed SYNTH_TEE_DIST_M away
    in the direction away from the course centre.

    Returns list of (tee_pos, green_pos, source) where source is
    'matched' or 'synthesized'.
    """
    try:
        from pyproj import Transformer
        t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        t_to_wgs = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)
    except Exception as e:
        log.warning(f"Pyproj unavailable for corridor pairing: {e}")
        return []

    # Project tees and greens to ITM (metres)
    tees_itm: List[Tuple[float, float, dict]] = []
    for t in tees:
        try:
            tx, ty = t_to_itm.transform(t["lon"], t["lat"])
            tees_itm.append((tx, ty, t))
        except Exception:
            pass

    greens_itm: List[Tuple[float, float, dict]] = []
    for g in greens:
        try:
            gx, gy = t_to_itm.transform(g["lon"], g["lat"])
            greens_itm.append((gx, gy, g))
        except Exception:
            pass

    cx, cy = t_to_itm.transform(course_center[0], course_center[1])

    pairs: List[Tuple[dict, dict, str]] = []
    assigned_tees: set = set()

    for gx, gy, green in greens_itm:
        # Find nearest unassigned tee within MAX_TEE_MATCH_M
        best_ti   = None
        best_dist = float("inf")
        for ti, (tx, ty, _tee) in enumerate(tees_itm):
            if ti in assigned_tees:
                continue
            d = math.sqrt((gx - tx) ** 2 + (gy - ty) ** 2)
            if d < best_dist and d <= MAX_TEE_MATCH_M:
                best_dist = d
                best_ti   = ti

        if best_ti is not None:
            assigned_tees.add(best_ti)
            _, _, tee = tees_itm[best_ti]
            pairs.append((tee, green, "matched"))
        else:
            # Synthesize: place tee away from course centre
            dx = gx - cx
            dy = gy - cy
            length = math.sqrt(dx * dx + dy * dy) or 1.0
            ux, uy = dx / length, dy / length

            syn_x = gx + ux * SYNTH_TEE_DIST_M
            syn_y = gy + uy * SYNTH_TEE_DIST_M
            try:
                syn_lon, syn_lat = t_to_wgs.transform(syn_x, syn_y)
                tee_dict = {
                    "lon":    float(syn_lon),
                    "lat":    float(syn_lat),
                    "source": "synthesized",
                }
                pairs.append((tee_dict, green, "synthesized"))
            except Exception:
                pass

    return pairs


# ── Corridor computation ───────────────────────────────────────────────────────

def _compute_corridor(
    tee_pos: dict,
    green_pos: dict,
    mosaic_arr: np.ndarray,
    transform_params: tuple,
    grass_mask: np.ndarray,
    tree_mask: np.ndarray,
) -> Optional[dict]:
    """
    Compute playable corridor polygon + centreline between tee and green.

    1. Define parallelogram strip ±CORRIDOR_HALF_W_M either side of axis
    2. Intersect with playable (grass & ~tree) mask
    3. Morphological closing to fill tree-gap fragments
    4. Largest contour = corridor polygon (WGS84 GeoJSON)
    5. Skeletonize → centreline sorted along tee→green axis

    Returns {"polygon": GeoJSON, "centerline": GeoJSON} or None.
    """
    try:
        import cv2
        from shapely.geometry import Polygon as SPoly, LineString as SLine, mapping

        west, north, lon_per_px, lat_per_px = transform_params
        h_px, w_px = mosaic_arr.shape[:2]

        def _wgs_to_px(lon: float, lat: float) -> Tuple[int, int]:
            return (
                max(0, min(w_px - 1, int((lon - west)  / lon_per_px))),
                max(0, min(h_px - 1, int((north - lat) / lat_per_px))),
            )

        tee_px   = _wgs_to_px(tee_pos["lon"],  tee_pos["lat"])
        green_px = _wgs_to_px(green_pos["lon"], green_pos["lat"])

        dx = green_px[0] - tee_px[0]
        dy = green_px[1] - tee_px[1]
        seg_len = math.sqrt(dx * dx + dy * dy)
        if seg_len < 4:
            return None

        # Metres per pixel estimate
        m_per_px = (lon_per_px * 111320 * math.cos(math.radians(tee_pos["lat"])) +
                    lat_per_px * 111320) / 2.0
        if m_per_px < 0.1:
            return None
        half_w_px = max(int(CORRIDOR_HALF_W_M / m_per_px), 8)

        # Perpendicular unit vector
        px = (-dy / seg_len, dx / seg_len)

        # Build corridor strip mask (parallelogram)
        strip_pts = np.array([
            (int(tee_px[0]   + px[0] * half_w_px), int(tee_px[1]   + px[1] * half_w_px)),
            (int(green_px[0] + px[0] * half_w_px), int(green_px[1] + px[1] * half_w_px)),
            (int(green_px[0] - px[0] * half_w_px), int(green_px[1] - px[1] * half_w_px)),
            (int(tee_px[0]   - px[0] * half_w_px), int(tee_px[1]   - px[1] * half_w_px)),
        ], dtype=np.int32)

        corridor_strip = np.zeros((h_px, w_px), dtype=np.uint8)
        cv2.fillPoly(corridor_strip, [strip_pts], 255)

        # Playable = grass AND NOT tree, clipped to corridor strip
        playable = cv2.bitwise_and(
            grass_mask,
            cv2.bitwise_not(tree_mask),
        )
        playable = cv2.bitwise_and(playable, corridor_strip)

        # Morphological closing to bridge small tree-shadow gaps
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        playable = cv2.morphologyEx(playable, cv2.MORPH_CLOSE, k, iterations=2)

        # Fall back to full strip if not enough grass detected
        strip_area    = float(cv2.countNonZero(corridor_strip))
        playable_area = float(cv2.countNonZero(playable))
        if strip_area < 10 or playable_area / strip_area < MIN_GRASS_FRAC:
            log.debug(
                f"Corridor: low grass fraction "
                f"({playable_area/max(strip_area,1):.2f}) — using full strip"
            )
            playable = corridor_strip

        # Take largest contour as the corridor polygon
        contours, _ = cv2.findContours(
            playable, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None
        best_c = max(contours, key=cv2.contourArea)

        # Build filled corridor mask for skeletonization
        corr_filled = np.zeros((h_px, w_px), dtype=np.uint8)
        cv2.drawContours(corr_filled, [best_c], -1, 255, thickness=cv2.FILLED)

        # Skeletonize → sorted centreline waypoints
        centerline_pts = _skeletonize_to_line(corr_filled, tee_px, green_px)

        # Convert contour → WGS84 polygon
        def _px_to_wgs(px_x: int, px_y: int) -> Tuple[float, float]:
            return (west + px_x * lon_per_px, north - px_y * lat_per_px)

        pts = best_c.squeeze()
        if pts.ndim != 2 or len(pts) < 3:
            return None
        poly_coords = [_px_to_wgs(int(p[0]), int(p[1])) for p in pts]

        corr_poly = SPoly(poly_coords)
        if not corr_poly.is_valid:
            corr_poly = corr_poly.buffer(0)
        if corr_poly.is_empty:
            return None

        result: dict = {"polygon": mapping(corr_poly)}
        if centerline_pts and len(centerline_pts) >= 2:
            line_coords = [_px_to_wgs(p[0], p[1]) for p in centerline_pts]
            result["centerline"] = mapping(SLine(line_coords))

        return result

    except Exception as e:
        log.debug(f"_compute_corridor failed: {e}")
        return None


def _skeletonize_to_line(
    mask: np.ndarray,
    tee_px: Tuple[int, int],
    green_px: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Skeletonize a binary mask and return waypoints sorted along the
    tee→green axis.  Returns [(x, y), ...] or the straight-line fallback.
    """
    try:
        from skimage.morphology import skeletonize as ski_skel

        skel = ski_skel(mask > 0).astype(np.uint8)
        rows, cols = np.where(skel)
        if len(rows) == 0:
            return [tee_px, green_px]

        pts_xy = list(zip(cols.tolist(), rows.tolist()))  # (x, y)

        # Sort by projection onto tee→green axis
        dx = green_px[0] - tee_px[0]
        dy = green_px[1] - tee_px[1]
        length = math.sqrt(dx * dx + dy * dy) or 1.0

        def _proj(p: Tuple[int, int]) -> float:
            return (p[0] - tee_px[0]) * dx / length + (p[1] - tee_px[1]) * dy / length

        pts_xy.sort(key=_proj)

        # Thin to ≤50 waypoints to keep GeoJSON small
        step = max(1, len(pts_xy) // 50)
        return pts_xy[::step]

    except ImportError:
        log.debug("scikit-image not available — using straight centreline")
        return [tee_px, green_px]
    except Exception:
        return [tee_px, green_px]


# ── Playable masks ─────────────────────────────────────────────────────────────

def _build_playable_masks(
    img_rgb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build grass mask and tree mask from the full satellite mosaic.
    Called once; both masks are reused for every corridor computation.

    Grass mask:  H:28-95, S:25-220, V:45-215 — covers fairway + rough
    Tree mask:   H:30-90, V:15-120, high local texture (stddev > 28)

    Returns (grass_mask, tree_mask) as uint8 arrays (255 = member, 0 = not).
    """
    try:
        import cv2

        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Grass: broad green range (includes fairway and rough)
        grass = cv2.inRange(
            hsv,
            np.array([28, 25, 45],  dtype=np.uint8),
            np.array([95, 220, 215], dtype=np.uint8),
        )

        # Local texture (approx std-dev of V channel, 7×7 window)
        v      = hsv[:, :, 2].astype(np.float32)
        v_blur = cv2.blur(v, (7, 7))
        v_sq   = cv2.blur(v * v, (7, 7))
        texture = np.sqrt(np.maximum(v_sq - v_blur * v_blur, 0.0))

        # Trees: dark green + high texture
        tree_hue = cv2.inRange(
            hsv,
            np.array([30, 20, 15],  dtype=np.uint8),
            np.array([90, 190, 120], dtype=np.uint8),
        )
        tree_tex  = (texture > 28).astype(np.uint8) * 255
        trees     = cv2.bitwise_and(tree_hue, tree_tex)

        # Clean masks
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grass = cv2.morphologyEx(grass, cv2.MORPH_CLOSE, k5, iterations=2)
        trees = cv2.morphologyEx(trees, cv2.MORPH_OPEN,  k5, iterations=1)

        return grass, trees

    except Exception as e:
        log.warning(f"Playable mask build failed: {e}")
        h, w = img_rgb.shape[:2]
        return (np.full((h, w), 255, dtype=np.uint8),
                np.zeros((h, w), dtype=np.uint8))


# ── Output ─────────────────────────────────────────────────────────────────────

def _write_corridor_fairways(holes: List[dict], output_dir: Path) -> None:
    """
    Write corridor_fairways.geojson.
    If fairways.geojson has fewer than 6 features (sparse OSM/vision result),
    also overwrite it with the corridor-derived polygons so that downstream
    modules (buildpack, companion UI, debug viz) use the corridor geometry.
    """
    from shapely.geometry import LineString, mapping
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    t_to_wgs = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

    features = []
    for h in holes:
        geom = h.get("fairway_corridor")
        if geom is None:
            # Fallback: buffer the tee→green straight line by 25m
            try:
                tp, gp = h["tee_position"], h["green_position"]
                line_itm = shp_transform(
                    t_to_itm.transform,
                    LineString([(tp["lon"], tp["lat"]), (gp["lon"], gp["lat"])]),
                )
                geom = mapping(shp_transform(t_to_wgs.transform, line_itm.buffer(25.0, cap_style=2)))
            except Exception:
                continue

        features.append({
            "type":     "Feature",
            "geometry": geom,
            "properties": {
                "hole_number": h["hole_number"],
                "type":        "fairway_corridor",
                "source":      h.get("routing_source", "corridor"),
                "distance_m":  h.get("distance_m"),
                "confidence":  h.get("confidence", 0.5),
            },
        })

    fc = {"type": "FeatureCollection", "features": features}
    (output_dir / "corridor_fairways.geojson").write_text(
        json.dumps(fc, indent=2), encoding="utf-8"
    )
    log.info(
        f"Corridor routing: {len(features)} corridor fairways → "
        f"corridor_fairways.geojson"
    )

    # Overwrite sparse fairways.geojson
    fw_path = output_dir / "fairways.geojson"
    try:
        existing_count = len(
            json.loads(fw_path.read_text(encoding="utf-8")).get("features", [])
        )
    except Exception:
        existing_count = 0

    if existing_count < 6:
        fw_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
        log.info(
            f"Corridor routing: fairways.geojson updated "
            f"({existing_count} → {len(features)} corridor-derived features)"
        )


# ── Utilities ─────────────────────────────────────────────────────────────────

def _load_mosaic(
    output_dir: Path,
) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
    """
    Load satellite_mosaic.jpg and reconstruct transform_params.

    Tries in order:
    1. transform_params field in vision_summary.json (added by this pipeline)
    2. Approximate from bbox_wgs84 in boundary.json + image dimensions
    """
    from PIL import Image

    mosaic_path = output_dir / "satellite_mosaic.jpg"
    if not mosaic_path.exists():
        return None, None

    try:
        img = np.array(Image.open(str(mosaic_path)).convert("RGB"), dtype=np.uint8)
    except Exception as e:
        log.warning(f"Corridor mosaic load failed: {e}")
        return None, None

    # Try vision_summary.json first
    summary_path = output_dir / "vision_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            tp = summary.get("transform_params")
            if tp and len(tp) == 4:
                return img, tuple(tp)
        except Exception:
            pass

    # Fall back to boundary.json
    boundary_path = output_dir / "boundary.json"
    if boundary_path.exists():
        try:
            bd   = json.loads(boundary_path.read_text(encoding="utf-8"))
            bbox = bd["bbox_wgs84"]
            h, w = img.shape[:2]
            tp = (
                float(bbox[0]),
                float(bbox[3]),
                (float(bbox[2]) - float(bbox[0])) / max(w, 1),
                (float(bbox[3]) - float(bbox[1])) / max(h, 1),
            )
            return img, tp
        except Exception:
            pass

    log.warning("Corridor routing: could not reconstruct transform_params")
    return img, None


def _course_center(greens: List[dict]) -> Tuple[float, float]:
    lons = [g["lon"] for g in greens if "lon" in g]
    lats = [g["lat"] for g in greens if "lat" in g]
    if not lons:
        return (0.0, 0.0)
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(min(a, 1.0)))


def _estimate_par(dist_m: float) -> int:
    yards = dist_m * 1.09361
    if yards < 250:
        return 3
    elif yards < 470:
        return 4
    return 5

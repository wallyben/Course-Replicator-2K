"""
routing.py — Golf course routing reconstruction.

UPGRADE 4: Reconstruct 18-hole routing from detected tees and greens.

Algorithm:
  1. Cluster tee positions (separate holes from multi-tee boxes)
  2. Cluster green polygons
  3. Pair tees → greens using nearest-valid-path heuristic
  4. Resolve routing order (sequential, non-overlapping)
  5. Generate fairway corridor polygons per hole

Output:
  - holes.geojson — 18-hole routing with tee/green/fairway per hole
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)

# Routing constraints
MIN_HOLE_LENGTH_M  = 50     # minimum tee-to-green distance
MAX_HOLE_LENGTH_M  = 650    # maximum (~710 yards)
MAX_HOLES          = 18
FAIRWAY_CORRIDOR_M = 40     # half-width of generated fairway corridor


# ─── Public API ───────────────────────────────────────────────────────────────

def reconstruct_routing(
    features_data: dict,
    output_dir: Path,
    osm_features_dir: Optional[Path] = None,
) -> dict:
    """
    Reconstruct 18-hole course routing from tee and green positions.

    Args:
        features_data:    Output from features.extract_features()
        output_dir:       Directory to write holes.geojson
        osm_features_dir: Optional directory with per-type GeoJSON from osm_features.py

    Returns:
        Routing dict with holes list and output path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tee and green positions from multiple sources
    tees_raw = _load_tee_positions(features_data, osm_features_dir)
    greens   = _load_green_polygons(features_data, osm_features_dir)

    # UPGRADE 1: Cluster multi-colour tee boxes (35m radius, keep backmost)
    tees = _cluster_tees_proper(tees_raw, greens, cluster_radius_m=35)
    log.info(
        f"Routing: {len(tees_raw)} raw tees → {len(tees)} clusters, "
        f"{len(greens)} greens found"
    )
    log.info(f"Tee clusters detected: {len(tees)}")

    if not tees and not greens:
        log.warning("No tee or green data available — cannot reconstruct routing")
        holes = _generate_placeholder_holes(features_data)
    elif not tees:
        log.warning("No tee data — using green centroids as tee estimates")
        holes = _route_from_greens_only(greens)
    elif not greens:
        log.warning("No green data — routing incomplete")
        holes = _route_from_tees_only(tees)
    else:
        # ── PRIMARY: Corridor routing (tee→corridor→green) ───────────────────
        # This is the most reliable method when fairway detection is sparse.
        # It anchors on greens (reliably detected) and synthesizes tees for
        # any greens that have no matching tee cluster within 550m.
        holes = _corridor_routing(tees, greens, output_dir)

        # ── FALLBACK 1: Multi-segment fairway graph ───────────────────────────
        if len(holes) < 9:
            log.info(
                f"Corridor routing yielded {len(holes)} holes — "
                f"trying fairway graph fallback"
            )
            fw_holes = _multi_segment_routing(osm_features_dir, greens, tees)
            if fw_holes and len(fw_holes) > len(holes):
                log.info(f"Fairway graph: {len(fw_holes)} holes (better than corridor)")
                holes = fw_holes

        # ── FALLBACK 2: Skeleton routing ──────────────────────────────────────
        if len(holes) < 9:
            log.info(
                f"Fairway graph yielded {len(holes)} holes — "
                f"trying skeleton fallback"
            )
            skeleton_holes = _skeleton_routing(osm_features_dir, greens)
            if skeleton_holes and len(skeleton_holes) >= 9:
                log.info(f"Skeleton fallback: {len(skeleton_holes)} holes")
                holes = skeleton_holes

        # ── FALLBACK 3: Pair-matching ─────────────────────────────────────────
        if len(holes) < 9:
            log.info(
                f"Pair-matching fallback from {len(tees)} tees, "
                f"{len(greens)} greens"
            )
            holes = _pair_tees_to_greens(tees, greens)

    # UPGRADE 3: Order holes sequentially (each tee near previous green)
    clubhouse = _find_clubhouse_pos(osm_features_dir)
    holes = _order_holes_sequentially(holes, clubhouse)

    # Add fairway corridors
    holes = _attach_fairway_corridors(holes, features_data, osm_features_dir)

    # Step 9: Validate holes (remove impossible lengths)
    holes_before = len(holes)
    holes = _validate_holes(holes, osm_features_dir)
    if len(holes) < holes_before:
        log.info(f"Routing validation: {holes_before} → {len(holes)} holes retained")

    # Write output
    holes_path = output_dir / "holes.geojson"
    _write_holes_geojson(holes, holes_path)
    log.info(f"Routing: {len(holes)} holes written to {holes_path.name}")

    # Write routing_debug.geojson (tee→green LineStrings per hole)
    try:
        _write_routing_debug_geojson(holes, output_dir)
    except Exception as e:
        log.debug(f"routing_debug.geojson export failed: {e}")

    # Write holes_metadata.json — authoritative routing summary.
    # NOTE: features.py writes osm_holes_metadata.json (list format) so there
    # is no filename collision.  This dict format is what QA and translation read.
    meta_path    = output_dir / "holes_metadata.json"
    meta_payload = json.dumps({
        "hole_count":       len(holes),
        "routing_method":   "inferred",
        "source":           "vision + osm",
        "has_inferred_tees": all(
            h.get("routing_source") in (
                "greens_only", "reconstructed", "skeleton", "placeholder",
                "corridor_matched", "corridor_synthesized",
            )
            for h in holes
        ),
    }, indent=2)
    meta_path.write_text(meta_payload, encoding="utf-8")
    n_bytes = meta_path.stat().st_size
    log.info(
        f"Routing: holes_metadata.json written — {len(holes)} holes, "
        f"{n_bytes} bytes, path={meta_path}"
    )

    return {
        "hole_count":  len(holes),
        "holes":       holes,
        "holes_path":  str(holes_path),
    }


# ─── Data loaders ─────────────────────────────────────────────────────────────

def _load_tee_positions(features_data: dict, osm_dir: Optional[Path]) -> List[dict]:
    """
    Load tee positions from features_data holes metadata and optional OSM GeoJSON.
    Returns list of {lon, lat, hole_number (if known), source}.
    """
    from shapely.geometry import shape

    tees = []

    # From holes_metadata (features.py)
    for hole in features_data.get("holes", []):
        tc = hole.get("tee_centroid")
        if tc and len(tc) == 2:
            tees.append({
                "lon":         tc[0],
                "lat":         tc[1],
                "hole_number": hole.get("hole_number"),
                "source":      "features_metadata",
            })

    # From OSM tees GeoJSON (osm_features.py)
    if osm_dir:
        tees_path = osm_dir / "tees.geojson"
        if tees_path.exists():
            feats = _load_geojson_features(tees_path)
            for f in feats:
                if not f.get("geometry"):
                    continue
                geom = shape(f["geometry"])
                c = geom.centroid
                tees.append({
                    "lon":         c.x,
                    "lat":         c.y,
                    "hole_number": f.get("properties", {}).get("ref"),
                    "source":      "osm_tees",
                })

    # Deduplicate tees that are within 5m of each other (same tee box)
    return _cluster_positions(tees, cluster_radius_m=8)


def _load_green_polygons(features_data: dict, osm_dir: Optional[Path]) -> List[dict]:
    """
    Load green polygons from features_data and optional OSM GeoJSON.
    Returns list of {lon, lat, polygon_wgs84, area_m2, source}.
    """
    from shapely.geometry import shape
    from shapely.ops import transform
    from pyproj import Transformer

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)

    greens = []

    # From holes_metadata (features.py)
    for hole in features_data.get("holes", []):
        gc = hole.get("green_centroid")
        if gc and len(gc) == 2:
            greens.append({
                "lon":         gc[0],
                "lat":         gc[1],
                "hole_number": hole.get("hole_number"),
                "source":      "features_metadata",
            })

    # From OSM greens GeoJSON
    if osm_dir:
        greens_path = osm_dir / "greens.geojson"
        if greens_path.exists():
            feats = _load_geojson_features(greens_path)
            for f in feats:
                if not f.get("geometry"):
                    continue
                try:
                    geom = shape(f["geometry"])
                    c = geom.centroid
                    geom_itm = transform(t_to_itm.transform, geom)
                    greens.append({
                        "lon":         c.x,
                        "lat":         c.y,
                        "area_m2":     geom_itm.area,
                        "hole_number": f.get("properties", {}).get("ref"),
                        "source":      "osm_greens",
                        "geometry":    f["geometry"],
                    })
                except Exception:
                    continue

    return _cluster_positions(greens, cluster_radius_m=15)


# ─── Pairing algorithm ────────────────────────────────────────────────────────

def _pair_tees_to_greens(tees: List[dict], greens: List[dict]) -> List[dict]:
    """
    Pair tees to greens using nearest-valid-path heuristic.

    Algorithm:
      1. For each tee, find candidate greens within hole length bounds
      2. Assign closest unassigned green
      3. Assign hole numbers sequentially
      4. If tee count > green count or vice versa, handle gracefully
    """
    from scipy.spatial.distance import cdist

    tee_coords  = np.array([[t["lat"], t["lon"]] for t in tees])
    green_coords = np.array([[g["lat"], g["lon"]] for g in greens])

    # Distance matrix in metres
    dist_matrix = _haversine_matrix(tee_coords, green_coords)

    assigned_greens = set()
    holes = []
    hole_num = 1

    # Sort tees by latitude (north → south) for initial ordering heuristic
    tee_order = np.argsort([t["lat"] for t in tees])[::-1]

    for ti in tee_order:
        if hole_num > MAX_HOLES:
            break

        dists = dist_matrix[ti, :]

        # Find nearest unassigned green within valid hole length range
        best_gi = None
        best_dist = float("inf")
        for gi in np.argsort(dists):
            if gi in assigned_greens:
                continue
            d = dists[gi]
            if MIN_HOLE_LENGTH_M <= d <= MAX_HOLE_LENGTH_M:
                best_gi = gi
                best_dist = d
                break
            elif d > MAX_HOLE_LENGTH_M:
                break  # sorted by distance, remaining are farther

        if best_gi is None:
            # Relax: take any nearest unassigned green
            for gi in np.argsort(dists):
                if gi not in assigned_greens:
                    best_gi = gi
                    best_dist = dists[gi]
                    break

        if best_gi is None:
            continue

        assigned_greens.add(best_gi)
        tee  = tees[ti]
        green = greens[best_gi]

        par = _estimate_par(best_dist)
        holes.append({
            "hole_number":         hole_num,
            "tee_position":        {"lon": tee["lon"],   "lat": tee["lat"]},
            "green_position":      {"lon": green["lon"],  "lat": green["lat"]},
            "distance_m":          round(best_dist, 1),
            "distance_yards":      round(best_dist * 1.09361, 0),
            "estimated_length_m":  round(best_dist, 1),
            "estimated_length_yd": round(best_dist * 1.09361, 0),
            "par":                 par,
            "routing_source":      "reconstructed",
            "confidence":          0.55,
            "source_mix":          f"{tee.get('source','unknown')}+{green.get('source','unknown')}",
        })
        hole_num += 1

    # Sort by sequential hole number
    holes.sort(key=lambda h: h["hole_number"])
    return holes


def _haversine_matrix(coords_a: np.ndarray, coords_b: np.ndarray) -> np.ndarray:
    """
    Compute haversine distance matrix in metres.
    coords: [[lat, lon], ...]
    """
    R = 6371000.0
    lat_a = np.radians(coords_a[:, 0])[:, np.newaxis]
    lon_a = np.radians(coords_a[:, 1])[:, np.newaxis]
    lat_b = np.radians(coords_b[:, 0])[np.newaxis, :]
    lon_b = np.radians(coords_b[:, 1])[np.newaxis, :]

    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def _estimate_par(distance_m: float) -> int:
    """Estimate par from tee-to-green distance."""
    yards = distance_m * 1.09361
    if yards < 250:
        return 3
    elif yards < 470:
        return 4
    else:
        return 5


# ─── Fairway corridors ────────────────────────────────────────────────────────

def _attach_fairway_corridors(
    holes: List[dict],
    features_data: dict,
    osm_dir: Optional[Path],
) -> List[dict]:
    """
    For each hole, either:
    - Attach existing OSM fairway that overlaps the tee→green line, or
    - Generate a synthetic fairway corridor polygon

    Adds "fairway_corridor" GeoJSON geometry to each hole dict.
    """
    from shapely.geometry import LineString, mapping
    from shapely.ops import transform, unary_union
    from pyproj import Transformer

    # Load all known fairways
    fairway_geoms = []
    if osm_dir:
        fw_path = osm_dir / "fairways.geojson"
        if fw_path.exists():
            from shapely.geometry import shape
            feats = _load_geojson_features(fw_path)
            for f in feats:
                if f.get("geometry"):
                    try:
                        fairway_geoms.append(shape(f["geometry"]))
                    except Exception:
                        pass

    for hole in holes:
        tee_pos   = hole["tee_position"]
        green_pos = hole["green_position"]
        if tee_pos is None or green_pos is None:
            hole["fairway_corridor"] = None
            continue
        line = LineString([
            (tee_pos["lon"], tee_pos["lat"]),
            (green_pos["lon"], green_pos["lat"]),
        ])

        # Find overlapping OSM fairway
        matched = None
        for fg in fairway_geoms:
            if fg.intersects(line.buffer(0.001)):  # ~100m buffer in degrees
                matched = fg
                break

        if matched is not None:
            hole["fairway_corridor"] = mapping(matched)
        else:
            # Generate synthetic corridor: buffer the tee→green line
            # Convert to ITM for metric buffering, then back
            try:
                t_to_itm  = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
                t_to_wgs  = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)
                from shapely.ops import transform as shp_transform
                line_itm  = shp_transform(t_to_itm.transform, line)
                corr_itm  = line_itm.buffer(FAIRWAY_CORRIDOR_M, cap_style=2)
                corr_wgs  = shp_transform(t_to_wgs.transform, corr_itm)
                hole["fairway_corridor"] = mapping(corr_wgs)
            except Exception as e:
                log.debug(f"Corridor generation failed for hole {hole['hole_number']}: {e}")
                hole["fairway_corridor"] = None

    return holes


# ─── Skeleton routing (Step 5) ────────────────────────────────────────────────

def _skeleton_routing(
    osm_dir: Optional[Path],
    greens: List[dict],
) -> List[dict]:
    """
    Attempt fairway skeleton-based routing.

    Algorithm:
      1. Load fairway polygons from OSM GeoJSON
      2. Rasterize each fairway polygon into a binary image
      3. Skeletonize to find the centreline
      4. Find branch endpoints as candidate tee positions
      5. Connect skeleton endpoints to the nearest unassigned green
      6. Generate hole list ordered by skeleton path

    Returns empty list on failure (caller falls back to pair matching).
    """
    if osm_dir is None:
        return []

    try:
        from shapely.geometry import shape, LineString, Point, mapping
        from shapely.ops import transform as shp_transform
        from pyproj import Transformer
        import numpy as np

        fw_path = osm_dir / "fairways.geojson"
        if not fw_path.exists():
            return []

        feats = _load_geojson_features(fw_path)
        if not feats:
            return []

        try:
            from skimage.morphology import skeletonize
        except ImportError:
            log.debug("scikit-image not available — skeleton routing skipped")
            return []

        t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        t_to_wgs = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

        holes = []
        assigned_greens = set()
        hole_num = 1

        for feat in feats:
            if hole_num > MAX_HOLES:
                break
            if not feat.get("geometry"):
                continue
            try:
                geom_wgs = shape(feat["geometry"])
                geom_itm = shp_transform(t_to_itm.transform, geom_wgs)
                bounds   = geom_itm.bounds
                w = max(int(bounds[2] - bounds[0]), 1)
                h = max(int(bounds[3] - bounds[1]), 1)

                # Skip extremely large or tiny fairways
                if w * h > 2_000_000 or w * h < 100:
                    continue

                # Build binary raster at 1m/px
                res = 2  # 2m per pixel for speed
                cols = max(w // res + 2, 4)
                rows = max(h // res + 2, 4)
                img  = np.zeros((rows, cols), dtype=np.uint8)

                # Rasterize: fill pixels inside fairway polygon
                from shapely.affinity import affine_transform
                minx, miny = bounds[0], bounds[1]
                scale = 1.0 / res
                # Simple bounding-box rasterization
                for r in range(rows):
                    for c in range(cols):
                        px = minx + c * res
                        py = miny + r * res
                        pt = Point(px, py)
                        if geom_itm.contains(pt):
                            img[r, c] = 1

                skel = skeletonize(img)
                skel_pts = list(zip(*np.where(skel)))
                if len(skel_pts) < 3:
                    continue

                # Find endpoints (pixels with only 1 neighbour in skeleton)
                endpoints = []
                for (r, c) in skel_pts:
                    neighbours = 0
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if (dr, dc) == (0, 0):
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < rows and 0 <= nc < cols and skel[nr, nc]:
                                neighbours += 1
                    if neighbours == 1:
                        # Convert back to ITM coords
                        px = minx + c * res
                        py = miny + r * res
                        endpoints.append((px, py))

                if not endpoints:
                    continue

                # Take the endpoint with minimum Y (south) as candidate tee
                tee_itm = min(endpoints, key=lambda p: p[1])
                tee_wgs = t_to_wgs.transform(tee_itm[0], tee_itm[1])

                # Find nearest unassigned green
                best_gi   = None
                best_dist = float("inf")
                for gi, g in enumerate(greens):
                    if gi in assigned_greens:
                        continue
                    d = _haversine(tee_wgs[0], tee_wgs[1], g["lon"], g["lat"])
                    if d < best_dist:
                        best_dist = d
                        best_gi   = gi

                if best_gi is None or not (MIN_HOLE_LENGTH_M <= best_dist <= MAX_HOLE_LENGTH_M * 1.5):
                    continue

                assigned_greens.add(best_gi)
                green = greens[best_gi]
                par   = _estimate_par(best_dist)
                holes.append({
                    "hole_number":    hole_num,
                    "tee_position":   {"lon": tee_wgs[0], "lat": tee_wgs[1]},
                    "green_position": {"lon": green["lon"], "lat": green["lat"]},
                    "distance_m":     round(best_dist, 1),
                    "distance_yards": round(best_dist * 1.09361, 0),
                    "par":            par,
                    "routing_source": "skeleton",
                })
                hole_num += 1

            except Exception as e:
                log.debug(f"Skeleton routing failed for fairway: {e}")
                continue

        if holes:
            holes.sort(key=lambda h: h["hole_number"])
            log.info(f"Skeleton routing: {len(holes)} holes from {len(feats)} fairways")
        return holes

    except Exception as e:
        log.debug(f"Skeleton routing aborted: {e}")
        return []


# ─── Routing validation (Step 9) ─────────────────────────────────────────────

def _validate_holes(holes: List[dict], osm_dir: Optional[Path]) -> List[dict]:
    """
    Validate each hole and reject physically implausible ones.

    Checks:
      1. Hole length must be between 80m and 700m
      2. Hole path should intersect a fairway (if OSM data available)
      3. Hole must have a defined green position

    Holes that fail hard checks are rejected.
    Holes that fail soft checks get a warning flag but are kept.
    """
    from shapely.geometry import LineString, Point

    # Load fairway geometries for intersection check
    fairway_union = None
    if osm_dir:
        try:
            fw_path = osm_dir / "fairways.geojson"
            if fw_path.exists():
                from shapely.geometry import shape
                from shapely.ops import unary_union
                feats = _load_geojson_features(fw_path)
                fws = [shape(f["geometry"]) for f in feats if f.get("geometry")]
                if fws:
                    fairway_union = unary_union(fws)
        except Exception:
            pass

    valid = []
    for hole in holes:
        tee   = hole.get("tee_position")
        green = hole.get("green_position")

        # Hard check: must have a green
        if green is None:
            log.debug(f"Hole {hole['hole_number']}: no green — rejected")
            continue

        # Hard check: distance must be plausible
        dist = hole.get("distance_m")
        if dist is not None and not (80 <= dist <= 700):
            log.debug(f"Hole {hole['hole_number']}: distance {dist:.0f}m out of range — rejected")
            continue

        # Soft check: path should intersect a fairway
        if fairway_union is not None and tee and green:
            try:
                line = LineString([
                    (tee["lon"], tee["lat"]),
                    (green["lon"], green["lat"]),
                ])
                if not fairway_union.intersects(line.buffer(0.0005)):
                    hole["routing_warning"] = "path_no_fairway_intersection"
                    log.debug(f"Hole {hole['hole_number']}: path does not intersect fairway (warning)")
            except Exception:
                pass

        valid.append(hole)

    return valid


# ─── Fallback generators ──────────────────────────────────────────────────────

def _route_from_greens_only(greens: List[dict]) -> List[dict]:
    """Generate routing using only green positions (no tee data)."""
    holes = []
    for i, green in enumerate(greens[:MAX_HOLES]):
        holes.append({
            "hole_number":    i + 1,
            "tee_position":   None,
            "green_position": {"lon": green["lon"], "lat": green["lat"]},
            "distance_m":     None,
            "distance_yards": None,
            "par":            None,
            "routing_source": "greens_only",
        })
    return holes


def _route_from_tees_only(tees: List[dict]) -> List[dict]:
    """Generate routing using only tee positions (no green data)."""
    holes = []
    for i, tee in enumerate(tees[:MAX_HOLES]):
        holes.append({
            "hole_number":    i + 1,
            "tee_position":   {"lon": tee["lon"], "lat": tee["lat"]},
            "green_position": None,
            "distance_m":     None,
            "distance_yards": None,
            "par":            None,
            "routing_source": "tees_only",
        })
    return holes


def _generate_placeholder_holes(features_data: dict) -> List[dict]:
    """Generate placeholder routing when no spatial data is available."""
    hole_count = features_data.get("hole_count", 18)
    holes = []
    for i in range(min(hole_count, MAX_HOLES)):
        holes.append({
            "hole_number":    i + 1,
            "tee_position":   None,
            "green_position": None,
            "distance_m":     None,
            "distance_yards": None,
            "par":            4,
            "routing_source": "placeholder",
        })
    return holes


# ─── Clustering ───────────────────────────────────────────────────────────────

def _cluster_positions(positions: List[dict], cluster_radius_m: float) -> List[dict]:
    """
    Remove duplicate positions within cluster_radius_m of each other.
    Keeps the first occurrence (highest priority source).
    """
    if not positions:
        return []

    kept = [positions[0]]
    for pos in positions[1:]:
        is_dup = False
        for k in kept:
            d = _haversine(k["lon"], k["lat"], pos["lon"], pos["lat"])
            if d < cluster_radius_m:
                is_dup = True
                break
        if not is_dup:
            kept.append(pos)
    return kept


def _cluster_tees_proper(
    tees: List[dict],
    greens: List[dict],
    cluster_radius_m: float = 35.0,
) -> List[dict]:
    """
    UPGRADE 1 — Cluster multi-colour tee boxes (white/yellow/blue/red) that
    belong to the same hole.

    Algorithm:
      - Union-Find: group tees within cluster_radius_m of each other
      - Within each cluster, keep the tee furthest from the green centroid
        (backmost / championship tee)
      - Attach tee_cluster_id and cluster_size metadata

    Args:
        tees:             Raw tee list from _load_tee_positions
        greens:           Green list (used to compute green centroid reference)
        cluster_radius_m: Radius to group same-hole tees (default 35m)

    Returns:
        Deduplicated tee list — one tee per cluster.
    """
    if not tees:
        return []

    # Green centroid as reference for "backmost" direction
    if greens:
        gc_lon = sum(g["lon"] for g in greens) / len(greens)
        gc_lat = sum(g["lat"] for g in greens) / len(greens)
    else:
        gc_lon, gc_lat = tees[0]["lon"], tees[0]["lat"]

    n = len(tees)
    parent = list(range(n))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Build clusters
    for i in range(n):
        for j in range(i + 1, n):
            d = _haversine(tees[i]["lon"], tees[i]["lat"],
                           tees[j]["lon"], tees[j]["lat"])
            if d <= cluster_radius_m:
                pi, pj = _find(i), _find(j)
                if pi != pj:
                    parent[pi] = pj

    comps: Dict[int, List[int]] = {}
    for i in range(n):
        comps.setdefault(_find(i), []).append(i)

    kept = []
    for cid, members in enumerate(comps.values()):
        # Back tee = furthest from green centroid
        best_idx = max(
            members,
            key=lambda i: _haversine(tees[i]["lon"], tees[i]["lat"], gc_lon, gc_lat),
        )
        t = dict(tees[best_idx])
        t["tee_cluster_id"] = cid + 1
        t["cluster_size"]   = len(members)
        kept.append(t)

    return kept


# ─── Corridor routing (PRIMARY) ───────────────────────────────────────────────

def _corridor_routing(
    tees: List[dict],
    greens: List[dict],
    output_dir: Path,
) -> List[dict]:
    """
    Primary routing method: green-anchored corridor reconstruction.

    Delegates to pipeline.hole_corridor.reconstruct_corridors().
    This approach does NOT depend on fairway detection — it works directly
    from tee clusters + green centroids + satellite mosaic.

    For each of the 18 greens:
      - find nearest unassigned tee cluster ≤550m
      - synthesize a tee if none found (placed 130m away from course centre)
      - compute playable corridor between tee and green
      - validate length (80–700m)

    Returns hole list compatible with routing.py format, or [] on failure.
    """
    try:
        from pipeline.hole_corridor import reconstruct_corridors
        holes = reconstruct_corridors(tees, greens, output_dir)
        if holes:
            log.info(f"Corridor routing (primary): {len(holes)} holes")
        return holes
    except Exception as e:
        log.warning(f"Corridor routing failed ({e}) — falling back to fairway graph")
        return []


def _multi_segment_routing(
    osm_dir: Optional[Path],
    greens: List[dict],
    tees: List[dict],
) -> List[dict]:
    """
    UPGRADE 2 — Route holes using a fairway adjacency graph.

    Algorithm:
      1. Project all OSM fairway polygons to ITM (metres)
      2. Build adjacency: fairways within 25m → connected edge
      3. Union-Find connected components (each = one hole's fairway chain)
      4. For each component:
           a. Tee-end estimate: midpoint of southernmost boundary
           b. Refine: use nearest known tee if within 150m
           c. Pair to nearest unassigned green (component edge ≤150m away,
              tee-to-green distance within hole length bounds)
      5. Log "Fairway graph nodes/edges/Multi-segment routing holes"
      6. Return hole list; fall back to [] on any unrecoverable error.
    """
    if osm_dir is None:
        return []

    try:
        from shapely.geometry import shape, Point
        from shapely.ops import transform as shp_transform, unary_union
        from pyproj import Transformer

        fw_path = osm_dir / "fairways.geojson"
        if not fw_path.exists():
            return []
        feats = _load_geojson_features(fw_path)
        if not feats:
            return []

        t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
        t_to_wgs = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)

        # Project fairways to ITM
        fw_itm: List = []
        for feat in feats:
            if not feat.get("geometry"):
                continue
            try:
                g = shp_transform(t_to_itm.transform, shape(feat["geometry"]))
                if g.is_valid and not g.is_empty:
                    fw_itm.append(g)
            except Exception:
                pass

        if not fw_itm:
            return []

        n = len(fw_itm)
        log.info(f"Fairway graph nodes: {n}")

        # Build adjacency at 50m threshold (was 25m — too sparse for tree-separated fairways)
        ADJACENCY_M = 50.0
        edges: List[Tuple[int, int]] = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    if fw_itm[i].distance(fw_itm[j]) <= ADJACENCY_M:
                        edges.append((i, j))
                except Exception:
                    pass
        log.info(f"Fairway graph edges: {len(edges)} (adjacency threshold: {ADJACENCY_M}m)")

        # Union-Find
        parent = list(range(n))

        def _uf_find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in edges:
            pi, pj = _uf_find(i), _uf_find(j)
            if pi != pj:
                parent[pi] = pj

        comps: Dict[int, List[int]] = {}
        for i in range(n):
            comps.setdefault(_uf_find(i), []).append(i)

        # Pre-project tees and greens to ITM for fast distance queries
        tees_itm: List[Tuple[float, float, dict]] = []
        for t in tees:
            try:
                tx, ty = t_to_itm.transform(t["lon"], t["lat"])
                tees_itm.append((tx, ty, t))
            except Exception:
                pass

        greens_itm: List[Optional[Tuple[float, float, dict]]] = []
        for g in greens:
            try:
                gx, gy = t_to_itm.transform(g["lon"], g["lat"])
                greens_itm.append((gx, gy, g))
            except Exception:
                greens_itm.append(None)

        assigned_greens: set = set()
        holes: List[dict] = []
        hole_num = 1

        for comp_indices in comps.values():
            if hole_num > MAX_HOLES:
                break

            # Union fairways in this component
            try:
                comp_geom = unary_union([fw_itm[i] for i in comp_indices])
            except Exception:
                continue
            if comp_geom.is_empty or comp_geom.area < 500:
                continue

            bounds = comp_geom.bounds  # (minx, miny, maxx, maxy)

            # Tee-end estimate: midpoint of southernmost boundary
            tee_est_x = (bounds[0] + bounds[2]) / 2.0
            tee_est_y = bounds[1]
            tee_pos_itm = (tee_est_x, tee_est_y)
            tee_wgs_est = t_to_wgs.transform(tee_est_x, tee_est_y)

            # Refine with nearest known tee if within 150m
            best_tee_dist = float("inf")
            tee_lon, tee_lat = float(tee_wgs_est[0]), float(tee_wgs_est[1])
            for tx, ty, t in tees_itm:
                d = math.sqrt((tx - tee_est_x) ** 2 + (ty - tee_est_y) ** 2)
                if d < best_tee_dist:
                    best_tee_dist = d
                    if d < 150:
                        tee_pos_itm = (tx, ty)
                        tee_lon, tee_lat = t["lon"], t["lat"]

            # Find nearest unassigned green with a valid hole distance
            best_gi   = None
            best_dist = float("inf")
            for gi, gdata in enumerate(greens_itm):
                if gi in assigned_greens or gdata is None:
                    continue
                gx, gy, g = gdata
                try:
                    # Green must be near the component boundary
                    if comp_geom.distance(Point(gx, gy)) > 150:
                        continue
                    tx_u, ty_u = tee_pos_itm
                    full_dist = math.sqrt((gx - tx_u) ** 2 + (gy - ty_u) ** 2)
                    if (MIN_HOLE_LENGTH_M <= full_dist <= MAX_HOLE_LENGTH_M * 1.5
                            and full_dist < best_dist):
                        best_dist = full_dist
                        best_gi   = gi
                except Exception:
                    pass

            if best_gi is None:
                continue

            assigned_greens.add(best_gi)
            _, _, green = greens_itm[best_gi]

            # Confidence: higher when a real tee was found nearby
            tee_confidence = 0.75 if best_tee_dist < 150 else 0.55
            green_source   = green.get("source", "unknown")
            tee_source     = tees[0].get("source", "unknown") if tees else "unknown"
            source_mix     = f"fairway_graph+{green_source}"

            holes.append({
                "hole_number":         hole_num,
                "tee_position":        {"lon": tee_lon, "lat": tee_lat},
                "green_position":      {"lon": green["lon"], "lat": green["lat"]},
                "distance_m":          round(best_dist, 1),
                "distance_yards":      round(best_dist * 1.09361, 0),
                "estimated_length_m":  round(best_dist, 1),
                "estimated_length_yd": round(best_dist * 1.09361, 0),
                "par":                 _estimate_par(best_dist),
                "routing_source":      "multi_segment",
                "fairway_segments":    len(comp_indices),
                "confidence":          round(tee_confidence, 2),
                "source_mix":          source_mix,
            })
            hole_num += 1

        log.info(f"Multi-segment routing holes: {len(holes)}")

        # Write fairway adjacency graph as GeoJSON for debug_viz ──────────────
        if osm_dir and edges and t_to_wgs:
            try:
                _write_fairway_graph_geojson(fw_itm, edges, t_to_wgs, osm_dir)
            except Exception as ge:
                log.debug(f"fairway_graph.geojson write failed: {ge}")

        return holes

    except Exception as e:
        log.warning(f"Multi-segment routing failed: {e}")
        return []


def _order_holes_sequentially(
    holes: List[dict],
    clubhouse_pos: Optional[Tuple[float, float]] = None,
) -> List[dict]:
    """
    UPGRADE 3 — Re-order holes so each hole's tee is nearest to the
    previous hole's green.

    Start-hole selection (in priority order):
      1. Hole whose tee is nearest to clubhouse_pos (if provided)
      2. Northernmost tee (latitude proxy for traditional first tee)

    Holes without tee+green positions are appended at the end unchanged.
    All holes are renumbered 1–N.
    """
    positioned   = [h for h in holes
                    if h.get("tee_position") and h.get("green_position")]
    unpositioned = [h for h in holes
                    if not (h.get("tee_position") and h.get("green_position"))]

    if len(positioned) <= 1:
        for i, h in enumerate(holes):
            h["hole_number"] = i + 1
        return holes

    if clubhouse_pos:
        ch_lon, ch_lat = clubhouse_pos
        start_idx = min(
            range(len(positioned)),
            key=lambda i: _haversine(
                ch_lon, ch_lat,
                positioned[i]["tee_position"]["lon"],
                positioned[i]["tee_position"]["lat"],
            ),
        )
    else:
        # Northernmost tee as first-hole proxy
        start_idx = max(
            range(len(positioned)),
            key=lambda i: positioned[i]["tee_position"]["lat"],
        )

    ordered: List[dict] = [positioned[start_idx]]
    remaining = [h for i, h in enumerate(positioned) if i != start_idx]

    while remaining:
        prev_green = ordered[-1]["green_position"]
        best_i = min(
            range(len(remaining)),
            key=lambda i: _haversine(
                prev_green["lon"], prev_green["lat"],
                remaining[i]["tee_position"]["lon"],
                remaining[i]["tee_position"]["lat"],
            ),
        )
        ordered.append(remaining[best_i])
        remaining.pop(best_i)

    all_ordered = ordered + unpositioned
    for i, h in enumerate(all_ordered):
        h["hole_number"] = i + 1

    return all_ordered


def _find_clubhouse_pos(
    osm_dir: Optional[Path],
) -> Optional[Tuple[float, float]]:
    """
    Try to find the clubhouse location from OSM GeoJSON files.
    Returns (lon, lat) or None.
    """
    if osm_dir is None:
        return None
    for fname in ("features.geojson", "paths.geojson"):
        fpath = osm_dir / fname
        if not fpath.exists():
            continue
        try:
            for f in _load_geojson_features(fpath):
                props = f.get("properties", {})
                if (props.get("amenity") == "club_house"
                        or props.get("golf") == "clubhouse"):
                    geom = f.get("geometry", {})
                    if geom.get("type") == "Point":
                        c = geom.get("coordinates", [])
                        if len(c) >= 2:
                            return (float(c[0]), float(c[1]))
        except Exception:
            pass
    return None


def _haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Haversine distance in metres."""
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Debug GeoJSON exports ────────────────────────────────────────────────────

def _write_fairway_graph_geojson(
    fw_itm: list,
    edges: List[Tuple[int, int]],
    t_to_wgs,
    output_dir: Path,
) -> None:
    """
    Write fairway_graph.geojson — each edge in the fairway adjacency graph
    as a LineString connecting the centroids of the two adjacent fairways.

    Args:
        fw_itm:     List of Shapely fairway polygons in ITM coordinates
        edges:      List of (i, j) index pairs from the adjacency graph
        t_to_wgs:   PyProj Transformer ITM→WGS84
        output_dir: Directory to write fairway_graph.geojson
    """
    features = []
    for i, j in edges:
        try:
            ci = fw_itm[i].centroid
            cj = fw_itm[j].centroid
            lon1, lat1 = t_to_wgs.transform(ci.x, ci.y)
            lon2, lat2 = t_to_wgs.transform(cj.x, cj.y)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[lon1, lat1], [lon2, lat2]],
                },
                "properties": {
                    "from_fairway": i,
                    "to_fairway":   j,
                    "type":         "fairway_adjacency",
                },
            })
        except Exception:
            continue

    out_path = output_dir / "fairway_graph.geojson"
    out_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    log.info(f"fairway_graph.geojson: {len(features)} edges → {out_path.name}")


def _write_routing_debug_geojson(holes: List[dict], output_dir: Path) -> None:
    """
    Write routing_debug.geojson — per-hole tee-to-green LineStrings.

    This file contains one LineString feature per hole, from tee_position
    to green_position, with hole metadata as properties.  Used by the
    companion app and debug_viz to visualise routing.
    """
    features = []
    for h in holes:
        tee   = h.get("tee_position")   or {}
        green = h.get("green_position") or {}
        t_lon, t_lat = tee.get("lon"), tee.get("lat")
        g_lon, g_lat = green.get("lon"), green.get("lat")
        if None in (t_lon, t_lat, g_lon, g_lat):
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": [[t_lon, t_lat], [g_lon, g_lat]],
            },
            "properties": {
                "hole_number":    h.get("hole_number"),
                "par":            h.get("par"),
                "distance_yards": h.get("distance_yards"),
                "routing_source": h.get("routing_source"),
                "type":           "hole_routing",
            },
        })

    out_path = output_dir / "routing_debug.geojson"
    out_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )
    log.info(
        f"routing_debug.geojson: {len(features)} holes → {out_path.name}"
    )


# ─── Output ───────────────────────────────────────────────────────────────────

def _write_holes_geojson(holes: List[dict], out_path: Path) -> None:
    """
    Write holes as a GeoJSON FeatureCollection.
    Each hole is a LineString from tee to green, with properties.
    """
    from shapely.geometry import LineString, Point, mapping

    features = []
    for hole in holes:
        tee   = hole.get("tee_position")
        green = hole.get("green_position")

        if tee and green:
            geom = mapping(LineString([
                (tee["lon"], tee["lat"]),
                (green["lon"], green["lat"]),
            ]))
        elif tee:
            geom = mapping(Point(tee["lon"], tee["lat"]))
        elif green:
            geom = mapping(Point(green["lon"], green["lat"]))
        else:
            geom = None

        props = {k: v for k, v in hole.items()
                 if k not in ("fairway_corridor",)}

        features.append({"type": "Feature", "geometry": geom, "properties": props})

        # Fairway corridor as separate feature
        fc = hole.get("fairway_corridor")
        if fc:
            features.append({
                "type":       "Feature",
                "geometry":   fc,
                "properties": {
                    "hole_number": hole["hole_number"],
                    "type":        "fairway_corridor",
                },
            })

    geojson = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def _load_geojson_features(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("features", [])
    except Exception:
        return []

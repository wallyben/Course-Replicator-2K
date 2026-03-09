"""
boundary.py — Course boundary resolution via OpenStreetMap Overpass API.

Resolves a golf course name to:
- Boundary polygon (GeoJSON)
- Bounding box with buffer
- Centre point
- Basic metadata (name, OSM ID, area m²)
"""

import json
import logging
import math
import time
from typing import Optional

import requests
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
import pyproj
from pyproj import Transformer

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)

# FIX 3: Bbox area constraints (hectares → m²)
_BBOX_AREA_MIN_M2 = 40  * 10_000   # 40 ha
_BBOX_AREA_MAX_M2 = 200 * 10_000   # 200 ha


# ─── Overpass query templates ─────────────────────────────────────────────────

def _overpass_query_by_name(name: str) -> str:
    """Build an Overpass query to find a golf course by name."""
    escaped = name.replace('"', '\\"')
    return f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
(
  way["leisure"="golf_course"]["name"~"{escaped}",i];
  relation["leisure"="golf_course"]["name"~"{escaped}",i];
  way["landuse"="golf_course"]["name"~"{escaped}",i];
  relation["landuse"="golf_course"]["name"~"{escaped}",i];
);
out body;
>;
out skel qt;
"""


def _overpass_query_relation_full(relation_id: int) -> str:
    """Fetch all nodes of a relation."""
    return f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
relation({relation_id});
out body;
>;
out skel qt;
"""


def _overpass_query_way_full(way_id: int) -> str:
    """Fetch all nodes of a way."""
    return f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
way({way_id});
out body;
>;
out skel qt;
"""


# ─── Geometry reconstruction from Overpass JSON ──────────────────────────────

def _reconstruct_nodes(elements: list) -> dict:
    """Build node-id → (lon, lat) lookup from Overpass elements."""
    return {
        el["id"]: (el["lon"], el["lat"])
        for el in elements
        if el["type"] == "node"
    }


def _reconstruct_way_coords(way_nodes: list, node_map: dict) -> list:
    """Convert a way's node-ref list to coordinate pairs."""
    coords = []
    for nid in way_nodes:
        if nid in node_map:
            coords.append(node_map[nid])
    return coords


def _ways_to_polygon(elements: list) -> Optional[object]:
    """Attempt to reconstruct boundary polygon from Overpass way elements."""
    from shapely.geometry import Polygon, MultiPolygon, LinearRing
    from shapely.ops import polygonize

    node_map = _reconstruct_nodes(elements)
    rings = []

    for el in elements:
        if el["type"] != "way":
            continue
        coords = _reconstruct_way_coords(el.get("nodes", []), node_map)
        if len(coords) >= 4:
            rings.append(coords)

    if not rings:
        return None

    # Try direct polygon (single closed way)
    if len(rings) == 1:
        try:
            poly = Polygon(rings[0])
            if poly.is_valid:
                return poly
            return poly.buffer(0)
        except Exception:
            return None

    # Multiple ways — try polygonize
    from shapely.geometry import LineString
    lines = [LineString(r) for r in rings]
    polys = list(polygonize(lines))
    if polys:
        merged = unary_union(polys)
        return merged

    return None


# ─── Main resolution function ─────────────────────────────────────────────────

def resolve_boundary(course_name: str) -> dict:
    """
    Resolve a golf course name to boundary data.

    Resolution strategy (in order):
      1. Check KNOWN_COURSES lookup in config (instant, no network)
      2. Try full name against OSM Overpass
      3. Try progressively shortened name variants (drop "Golf Club", "Golf", etc.)
      4. Try Ireland-wide area search with significant word only
      5. Raise with helpful error including --bbox suggestion

    Returns:
        {
            "name": str,
            "osm_id": int or None,
            "osm_type": str,
            "boundary_wgs84": dict,
            "bbox_wgs84": list,
            "bbox_buffered_itm": list,
            "centre_wgs84": list,
            "area_m2": float,
            "matched_name": str,
            "confidence": str,
        }
    """
    log.info(f"Resolving boundary for: {course_name!r}")

    # ── Strategy 1: KNOWN_COURSES lookup ────────────────────────────────────
    key = course_name.strip().lower()
    if key in config.KNOWN_COURSES:
        bbox = config.KNOWN_COURSES[key]
        log.info(f"Found in KNOWN_COURSES — using hardcoded bbox: {bbox}")
        return boundary_from_bbox(*bbox, course_name=course_name)

    # ── Strategy 2–4: OSM Overpass with name variants ────────────────────────
    name_variants = _build_name_variants(course_name)
    elements      = []

    for variant in name_variants:
        log.info(f"Trying OSM query: {variant!r}")
        query = _overpass_query_by_name(variant)
        resp  = _overpass_request(query)
        elements = resp.get("elements", [])
        candidates = [e for e in elements if e["type"] in ("relation", "way")]
        if candidates:
            log.info(f"Found {len(candidates)} candidate(s) with variant {variant!r}")
            break
        log.info(f"No results for variant {variant!r}")

    if not elements or not [e for e in elements if e["type"] in ("relation", "way")]:
        # Build a useful error with the known-courses hint
        _raise_not_found(course_name)

    # Find the best candidate
    candidates = [e for e in elements if e["type"] in ("relation", "way")]
    scored = []
    for el in candidates:
        tags    = el.get("tags", {})
        el_name = tags.get("name", "")
        score   = _name_similarity(course_name.lower(), el_name.lower())
        scored.append((score, el))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_el = scored[0]

    confidence   = "HIGH" if best_score > 0.7 else ("MEDIUM" if best_score > 0.4 else "LOW")
    matched_name = best_el.get("tags", {}).get("name", "Unknown")

    log.info(f"Best match: {matched_name!r} (score={best_score:.2f}, confidence={confidence})")

    # Reconstruct geometry
    polygon = _ways_to_polygon(elements)
    if polygon is None:
        osm_type = best_el["type"]
        osm_id   = best_el["id"]
        if osm_type == "relation":
            full_resp = _overpass_request(_overpass_query_relation_full(osm_id))
        else:
            full_resp = _overpass_request(_overpass_query_way_full(osm_id))
        polygon = _ways_to_polygon(full_resp.get("elements", []))

    if polygon is None:
        raise ValueError(
            f"Could not reconstruct boundary polygon for {matched_name!r}. "
            "The OSM boundary may be incomplete. Use --bbox to specify manually."
        )

    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    centroid   = polygon.centroid
    bbox_wgs84 = list(polygon.bounds)

    transformer_to_itm = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
    from shapely.ops import transform
    polygon_itm = transform(transformer_to_itm.transform, polygon)

    buf = config.BOUNDARY_BUFFER_M
    itm_bounds = polygon_itm.bounds
    bbox_buffered_itm = [
        itm_bounds[0] - buf,
        itm_bounds[1] - buf,
        itm_bounds[2] + buf,
        itm_bounds[3] + buf,
    ]

    return {
        "name":                course_name,
        "matched_name":        matched_name,
        "osm_id":              best_el["id"],
        "osm_type":            best_el["type"],
        "boundary_wgs84":      mapping(polygon),
        "bbox_wgs84":          bbox_wgs84,
        "bbox_buffered_itm":   bbox_buffered_itm,
        "centre_wgs84":        [centroid.x, centroid.y],
        "area_m2":             polygon_itm.area,
        "confidence":          confidence,
    }


# ─── Manual fallback ─────────────────────────────────────────────────────────

def boundary_from_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    course_name: str = "Manual Entry"
) -> dict:
    """
    Create boundary data from a manually entered bounding box (WGS84).
    Use this if OSM boundary resolution fails.

    FIX 3: Automatically refines bbox to 40–200 ha by querying OSM for the
    actual golf_course polygon, then falling back to fairway cluster sizing.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform
    from pyproj import Transformer

    # FIX 3: Attempt to refine oversized/undersized bbox via OSM
    refined = _refine_bbox(min_lon, min_lat, max_lon, max_lat, course_name)
    min_lon, min_lat, max_lon, max_lat = refined

    polygon = shapely_box(min_lon, min_lat, max_lon, max_lat)
    centroid = polygon.centroid

    transformer_to_itm = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
    polygon_itm = transform(transformer_to_itm.transform, polygon)
    itm_bounds = polygon_itm.bounds
    buf = config.BOUNDARY_BUFFER_M

    return {
        "name":                course_name,
        "matched_name":        course_name,
        "osm_id":              None,
        "osm_type":            "manual",
        "boundary_wgs84":      mapping(polygon),
        "bbox_wgs84":          [min_lon, min_lat, max_lon, max_lat],
        "bbox_buffered_itm":   [
            itm_bounds[0] - buf, itm_bounds[1] - buf,
            itm_bounds[2] + buf, itm_bounds[3] + buf,
        ],
        "centre_wgs84":        [centroid.x, centroid.y],
        "area_m2":             polygon_itm.area,
        "confidence":          "LOW",
    }


# ─── FIX 3: Bbox refinement ───────────────────────────────────────────────────

def _refine_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    course_name: str,
) -> tuple:
    """
    FIX 3: Refine bounding box to stay within 40–200 ha.

    Algorithm:
      1. Query OSM for golf_course polygon within the initial bbox
      2. If found → use polygon bounds (tightest, most accurate)
      3. Otherwise → query fairway cluster and expand 120%
      4. Clamp result to [40 ha, 200 ha] around centroid

    Returns:
        (min_lon, min_lat, max_lon, max_lat) refined WGS84 tuple
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform, unary_union
    from pyproj import Transformer

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)

    # Measure initial area
    init_poly_itm = transform(
        t_to_itm.transform,
        shapely_box(min_lon, min_lat, max_lon, max_lat)
    )
    init_area = init_poly_itm.area
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    log.debug(f"Initial bbox area: {init_area/10000:.1f} ha")

    # Step 1: Try OSM golf_course polygon within bbox
    try:
        osm_poly = _query_golf_course_polygon(min_lon, min_lat, max_lon, max_lat)
        if osm_poly is not None:
            bounds = osm_poly.bounds
            poly_itm = transform(t_to_itm.transform, osm_poly)
            area = poly_itm.area
            if _BBOX_AREA_MIN_M2 <= area <= _BBOX_AREA_MAX_M2:
                log.info(f"FIX 3: Using OSM golf_course polygon ({area/10000:.1f} ha)")
                return bounds  # (min_lon, min_lat, max_lon, max_lat)
            elif area < _BBOX_AREA_MIN_M2:
                log.debug(f"OSM polygon too small ({area/10000:.1f} ha), expanding")
                return _clamp_bbox_around_centroid(
                    bounds[0], bounds[1], bounds[2], bounds[3],
                    center_lon, center_lat, t_to_itm
                )
            else:
                log.debug(f"OSM polygon too large ({area/10000:.1f} ha), clamping")
                return _clamp_bbox_around_centroid(
                    bounds[0], bounds[1], bounds[2], bounds[3],
                    center_lon, center_lat, t_to_itm
                )
    except Exception as e:
        log.debug(f"FIX 3: OSM golf_course query failed: {e}")

    # Step 2: If initial bbox is already in range, keep it
    if _BBOX_AREA_MIN_M2 <= init_area <= _BBOX_AREA_MAX_M2:
        log.debug("FIX 3: Initial bbox already in range, keeping")
        return (min_lon, min_lat, max_lon, max_lat)

    # Step 3: Try fairway cluster sizing
    try:
        fairway_bbox = _query_fairway_cluster_bbox(min_lon, min_lat, max_lon, max_lat)
        if fairway_bbox is not None:
            fw_min_lon, fw_min_lat, fw_max_lon, fw_max_lat = fairway_bbox
            # Expand 120%
            fw_poly = shapely_box(fw_min_lon, fw_min_lat, fw_max_lon, fw_max_lat)
            fw_itm = transform(t_to_itm.transform, fw_poly)
            cx_itm, cy_itm = fw_itm.centroid.x, fw_itm.centroid.y
            hw = math.sqrt(fw_itm.area * 1.2) / 2
            log.info(f"FIX 3: Using 120% fairway cluster bbox ({fw_itm.area*1.2/10000:.1f} ha est.)")
            return _clamp_bbox_around_centroid(
                fw_min_lon, fw_min_lat, fw_max_lon, fw_max_lat,
                center_lon, center_lat, t_to_itm
            )
    except Exception as e:
        log.debug(f"FIX 3: Fairway cluster query failed: {e}")

    # Step 4: Clamp oversized initial bbox around centroid
    return _clamp_bbox_around_centroid(
        min_lon, min_lat, max_lon, max_lat,
        center_lon, center_lat, t_to_itm
    )


def _clamp_bbox_around_centroid(
    min_lon, min_lat, max_lon, max_lat,
    center_lon, center_lat, t_to_itm
) -> tuple:
    """
    Clamp bbox to [40–200 ha] centered on centroid, preserving aspect ratio.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform

    poly_itm = transform(
        t_to_itm.transform,
        shapely_box(min_lon, min_lat, max_lon, max_lat)
    )
    area = poly_itm.area

    if area < _BBOX_AREA_MIN_M2:
        target_area = _BBOX_AREA_MIN_M2
    elif area > _BBOX_AREA_MAX_M2:
        target_area = _BBOX_AREA_MAX_M2
    else:
        return (min_lon, min_lat, max_lon, max_lat)

    # Scale factor to achieve target area
    scale = math.sqrt(target_area / max(area, 1.0))
    half_lon = (max_lon - min_lon) / 2 * scale
    half_lat = (max_lat - min_lat) / 2 * scale

    # Maintain a minimum half-span of ~500m (~0.005°)
    half_lon = max(half_lon, 0.005)
    half_lat = max(half_lat, 0.004)

    new_min_lon = center_lon - half_lon
    new_max_lon = center_lon + half_lon
    new_min_lat = center_lat - half_lat
    new_max_lat = center_lat + half_lat

    new_area = transform(
        t_to_itm.transform,
        shapely_box(new_min_lon, new_min_lat, new_max_lon, new_max_lat)
    ).area
    log.info(f"FIX 3: Bbox clamped from {area/10000:.1f} ha → {new_area/10000:.1f} ha")
    return (new_min_lon, new_min_lat, new_max_lon, new_max_lat)


def _query_golf_course_polygon(min_lon, min_lat, max_lon, max_lat):
    """
    Query Overpass for a golf_course polygon within the bbox.
    Returns largest Shapely polygon found, or None.
    """
    from shapely.geometry import MultiPolygon

    query = f"""
[out:json][timeout:30];
(
  way["leisure"="golf_course"]({min_lat},{min_lon},{max_lat},{max_lon});
  relation["leisure"="golf_course"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out body;
>;
out skel qt;
"""
    resp = _overpass_request(query, retries=2)
    elements = resp.get("elements", [])
    if not elements:
        return None

    poly = _ways_to_polygon(elements)
    if poly is None:
        return None
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda p: p.area)
    return poly


def _query_fairway_cluster_bbox(min_lon, min_lat, max_lon, max_lat):
    """
    Query Overpass for fairway polygons within bbox.
    Returns bounding box of all fairways as (min_lon, min_lat, max_lon, max_lat), or None.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import unary_union

    query = f"""
[out:json][timeout:30];
(
  way["golf"="fairway"]({min_lat},{min_lon},{max_lat},{max_lon});
  way["golf"="green"]({min_lat},{min_lon},{max_lat},{max_lon});
);
out body;
>;
out skel qt;
"""
    resp = _overpass_request(query, retries=2)
    elements = resp.get("elements", [])
    if not elements:
        return None

    node_map = _reconstruct_nodes(elements)
    polygons = []
    for el in elements:
        if el["type"] != "way":
            continue
        coords = _reconstruct_way_coords(el.get("nodes", []), node_map)
        if len(coords) >= 4:
            try:
                from shapely.geometry import Polygon
                p = Polygon(coords)
                if p.is_valid and p.area > 0:
                    polygons.append(p)
            except Exception:
                continue

    if not polygons:
        return None

    merged = unary_union(polygons)
    bounds = merged.bounds  # (min_lon, min_lat, max_lon, max_lat)
    return bounds


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _overpass_request(query: str, retries: int = 3) -> dict:
    """Send an Overpass QL query and return parsed JSON."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                config.OVERPASS_URL,
                data={"data": query},
                timeout=config.OVERPASS_TIMEOUT + 10,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt < retries - 1:
                wait = 2 ** attempt
                log.warning(f"Overpass request failed (attempt {attempt+1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(f"Overpass API unavailable after {retries} attempts: {e}") from e
    return {}


def _name_similarity(a: str, b: str) -> float:
    """Simple token overlap similarity."""
    if not a or not b:
        return 0.0
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    # Strip common suffixes
    stop = {"golf", "club", "course", "links", "gc", "the"}
    tokens_a -= stop
    tokens_b -= stop
    if not tokens_a or not tokens_b:
        return 0.5
    intersection = tokens_a & tokens_b
    return len(intersection) / max(len(tokens_a), len(tokens_b))


def _build_name_variants(course_name: str) -> list:
    """
    Build a list of progressively shorter name variants to try against OSM.

    Example: "Old Conna Golf Club" →
      ["Old Conna Golf Club", "Old Conna Golf", "Old Conna", "Conna"]
    """
    variants = [course_name]
    suffixes_to_strip = [
        " Golf Club", " Golf Links", " Golf Course", " Golf & Country Club",
        " Golf", " Club", " Links", " Course",
    ]
    working = course_name
    for suffix in suffixes_to_strip:
        if working.lower().endswith(suffix.lower()):
            working = working[: len(working) - len(suffix)].strip()
            if working and working not in variants:
                variants.append(working)

    # Also try just the first significant word (for very specific searches)
    stop = {"golf", "club", "course", "links", "the", "old", "new", "royal"}
    words = [w for w in working.split() if w.lower() not in stop]
    if words and len(words[-1]) > 3:
        last_word = words[-1]
        if last_word not in variants and last_word != working:
            variants.append(last_word)

    return variants


def _raise_not_found(course_name: str) -> None:
    """Raise a helpful ValueError when no OSM match is found."""
    # Check if a nearby known course might help orient the user
    known_hint = ""
    key = course_name.strip().lower()
    # Suggest adding to KNOWN_COURSES
    known_hint = (
        f"\n\nIf this course is not in OpenStreetMap, add it to KNOWN_COURSES in config.py:\n"
        f'  "{key}": [min_lon, min_lat, max_lon, max_lat],\n'
        f"Or use --bbox directly:\n"
        f"  python scripts/run_pipeline.py \"{course_name}\" "
        f"--bbox min_lon,min_lat,max_lon,max_lat\n"
        f"(Find coordinates at openstreetmap.org — right-click → 'Show address')"
    )
    raise ValueError(
        f"No OSM golf course found for: {course_name!r}\n"
        f"Tried name variants but found nothing in OpenStreetMap."
        + known_hint
    )

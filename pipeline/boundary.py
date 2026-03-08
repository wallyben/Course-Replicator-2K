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

    Returns:
        {
            "name": str,
            "osm_id": int,
            "osm_type": str,          # "way" or "relation"
            "boundary_wgs84": dict,   # GeoJSON polygon
            "bbox_wgs84": list,       # [min_lon, min_lat, max_lon, max_lat]
            "bbox_buffered_itm": list, # [minx, miny, maxx, maxy] in EPSG:2157
            "centre_wgs84": list,     # [lon, lat]
            "area_m2": float,
            "matched_name": str,
            "confidence": str,        # HIGH / MEDIUM / LOW
        }
    """
    log.info(f"Resolving boundary for: {course_name!r}")

    query = _overpass_query_by_name(course_name)
    resp  = _overpass_request(query)
    elements = resp.get("elements", [])

    if not elements:
        raise ValueError(
            f"No OSM golf course found for: {course_name!r}. "
            "Try the exact club name (e.g. 'Lahinch Golf Club') or check OSM coverage."
        )

    # Find the best candidate: prefer relations, then ways
    candidates = [e for e in elements if e["type"] in ("relation", "way")]
    if not candidates:
        raise ValueError(f"OSM returned nodes only — no closed boundary found for {course_name!r}")

    # Score candidates — prefer those whose name closely matches
    scored = []
    for el in candidates:
        tags = el.get("tags", {})
        el_name = tags.get("name", "")
        score = _name_similarity(course_name.lower(), el_name.lower())
        scored.append((score, el))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_el = scored[0]

    confidence = "HIGH" if best_score > 0.7 else ("MEDIUM" if best_score > 0.4 else "LOW")
    matched_name = best_el.get("tags", {}).get("name", "Unknown")

    log.info(f"Best match: {matched_name!r} (score={best_score:.2f}, confidence={confidence})")

    # Reconstruct geometry
    polygon = _ways_to_polygon(elements)
    if polygon is None:
        # Try fetching the specific element in full
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
            "The OSM boundary may be incomplete. Try entering a manual bounding box."
        )

    # Ensure we have a single polygon (take largest if multi)
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    # Compute in WGS84
    centroid    = polygon.centroid
    bbox_wgs84  = list(polygon.bounds)  # [minx, miny, maxx, maxy] = [w, s, e, n]

    # Project to Irish Transverse Mercator for metric operations
    transformer_to_itm = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
    transformer_to_wgs = Transformer.from_crs(config.CRS_ITM, config.CRS_WGS84, always_xy=True)

    from shapely.ops import transform
    polygon_itm = transform(transformer_to_itm.transform, polygon)

    # Buffered bounding box in ITM
    buf = config.BOUNDARY_BUFFER_M
    itm_bounds = polygon_itm.bounds  # [minx, miny, maxx, maxy]
    bbox_buffered_itm = [
        itm_bounds[0] - buf,
        itm_bounds[1] - buf,
        itm_bounds[2] + buf,
        itm_bounds[3] + buf,
    ]

    area_m2 = polygon_itm.area

    return {
        "name":                course_name,
        "matched_name":        matched_name,
        "osm_id":              best_el["id"],
        "osm_type":            best_el["type"],
        "boundary_wgs84":      mapping(polygon),
        "bbox_wgs84":          bbox_wgs84,
        "bbox_buffered_itm":   bbox_buffered_itm,
        "centre_wgs84":        [centroid.x, centroid.y],
        "area_m2":             area_m2,
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
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform
    from pyproj import Transformer

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

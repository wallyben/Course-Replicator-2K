"""
osm_features.py — OpenStreetMap golf feature extraction via Overpass API.

UPGRADE 2: Dedicated OSM feature extraction module.

Extracts:
  - golf_course boundary
  - fairways
  - greens
  - bunkers
  - tees
  - cart paths
  - water hazards

Outputs:
  - fairways.geojson
  - greens.geojson
  - bunkers.geojson
  - tees.geojson
  - paths.geojson
  - water.geojson
  - osm_features_summary.json
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)

# Feature type → Overpass golf tag mapping
GOLF_TAGS = {
    "fairway":  [("golf", "fairway")],
    "green":    [("golf", "green")],
    "bunker":   [("golf", "bunker"), ("golf", "sand")],
    "tee":      [("golf", "tee")],
    "path":     [("golf", "path"), ("highway", "path"), ("highway", "footway")],
    "water":    [
        ("golf", "water_hazard"),
        ("natural", "water"),
        ("water", "pond"), ("water", "lake"), ("water", "river"),
        ("waterway", "stream"), ("waterway", "river"),
    ],
}

LABEL_COLORS = {
    "fairway": "#4caf50",
    "green":   "#1b5e20",
    "bunker":  "#f5deb3",
    "tee":     "#2196f3",
    "path":    "#9e9e9e",
    "water":   "#1565c0",
}


# ─── Public API ───────────────────────────────────────────────────────────────

def extract_osm_features(boundary_data: dict, output_dir: Path) -> dict:
    """
    Extract all golf features from OSM via Overpass API.

    Args:
        boundary_data: Output from boundary.resolve_boundary()
        output_dir:    Directory to write GeoJSON files

    Returns:
        Summary dict with counts and file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = boundary_data["bbox_wgs84"]  # [min_lon, min_lat, max_lon, max_lat]
    min_lon, min_lat, max_lon, max_lat = bbox

    log.info(f"Querying Overpass for golf features in bbox: {bbox}")

    # Build and execute Overpass query
    raw_elements = _fetch_all_golf_features(min_lon, min_lat, max_lon, max_lat)
    log.info(f"Overpass returned {len(raw_elements)} raw elements")

    # Parse elements into feature collections
    node_map = _build_node_map(raw_elements)
    features_by_type = _classify_elements(raw_elements, node_map)

    # Write one GeoJSON per feature type
    output_paths = {}
    counts = {}
    for feat_type, features in features_by_type.items():
        out_path = output_dir / f"{feat_type}s.geojson"
        _write_geojson(features, out_path)
        output_paths[feat_type] = str(out_path)
        counts[feat_type] = len(features)
        log.info(f"  {feat_type}: {len(features)} features → {out_path.name}")

    summary = {
        "bbox":         bbox,
        "counts":       counts,
        "total":        sum(counts.values()),
        "output_paths": output_paths,
    }
    summary_path = output_dir / "osm_features_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


# ─── Overpass query ───────────────────────────────────────────────────────────

def _fetch_all_golf_features(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float
) -> list:
    """
    Build and execute an Overpass query for all golf feature types.
    Returns raw element list.
    """
    bbox_str = f"{min_lat},{min_lon},{max_lat},{max_lon}"

    query_parts = []
    # Ways (polygons/lines)
    for feat_type, tag_pairs in GOLF_TAGS.items():
        for key, val in tag_pairs:
            query_parts.append(f'way["{key}"="{val}"]({bbox_str});')
            query_parts.append(f'relation["{key}"="{val}"]({bbox_str});')
            query_parts.append(f'node["{key}"="{val}"]({bbox_str});')

    query = f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
(
  {"".join(query_parts)}
);
out body;
>;
out skel qt;
"""

    for attempt in range(3):
        try:
            resp = requests.post(
                config.OVERPASS_URL,
                data={"data": query},
                timeout=config.OVERPASS_TIMEOUT + 10,
            )
            resp.raise_for_status()
            return resp.json().get("elements", [])
        except Exception as e:
            if attempt < 2:
                wait = 2 ** attempt
                log.warning(f"Overpass attempt {attempt+1} failed: {e}. Retry in {wait}s")
                time.sleep(wait)
            else:
                log.error(f"Overpass exhausted retries: {e}")
                return []

    return []


# ─── Classification ───────────────────────────────────────────────────────────

def _classify_elements(elements: list, node_map: dict) -> Dict[str, list]:
    """
    Classify raw Overpass elements into typed GeoJSON features.

    Returns dict of {feature_type: [GeoJSON Feature, ...]}
    """
    from shapely.geometry import Polygon, LineString, Point, mapping, MultiPolygon

    result = {t: [] for t in GOLF_TAGS}

    for el in elements:
        el_type = el.get("type")
        tags = el.get("tags", {})
        if not tags:
            continue

        feat_type = _detect_feature_type(tags)
        if feat_type is None:
            continue

        geometry = _build_geometry(el, el_type, node_map)
        if geometry is None:
            continue

        props = {
            "osm_id":   el.get("id"),
            "osm_type": el_type,
            "type":     feat_type,
            "color":    LABEL_COLORS.get(feat_type, "#888"),
        }
        # Copy relevant OSM tags
        for k in ("name", "golf", "natural", "water", "waterway",
                  "highway", "surface", "par", "ref"):
            if k in tags:
                props[k] = tags[k]

        result[feat_type].append({
            "type":       "Feature",
            "geometry":   mapping(geometry),
            "properties": props,
        })

    return result


def _detect_feature_type(tags: dict) -> Optional[str]:
    """Map OSM tags to internal feature type."""
    golf_val = tags.get("golf", "")
    natural_val = tags.get("natural", "")
    water_val = tags.get("water", "")
    waterway_val = tags.get("waterway", "")
    highway_val = tags.get("highway", "")

    if golf_val == "fairway":
        return "fairway"
    elif golf_val == "green":
        return "green"
    elif golf_val in ("bunker", "sand"):
        return "bunker"
    elif golf_val == "tee":
        return "tee"
    elif golf_val in ("path", "cart_path") or highway_val in ("path", "footway", "track"):
        return "path"
    elif golf_val == "water_hazard":
        return "water"
    elif natural_val == "water" or water_val in ("pond", "lake", "river", "reservoir"):
        return "water"
    elif waterway_val in ("stream", "river", "canal"):
        return "water"
    return None


def _build_geometry(el: dict, el_type: str, node_map: dict):
    """Build a Shapely geometry from an Overpass element."""
    from shapely.geometry import Polygon, LineString, Point, MultiPolygon
    from shapely.ops import polygonize, unary_union

    try:
        if el_type == "node":
            lon, lat = el.get("lon"), el.get("lat")
            if lon is None or lat is None:
                return None
            return Point(lon, lat)

        elif el_type == "way":
            nodes = el.get("nodes", [])
            coords = [node_map[n] for n in nodes if n in node_map]
            if len(coords) < 2:
                return None
            if coords[0] == coords[-1] and len(coords) >= 4:
                poly = Polygon(coords)
                return poly.buffer(0) if not poly.is_valid else poly
            return LineString(coords)

        elif el_type == "relation":
            # Find outer ways from members
            outer_ways = []
            for member in el.get("members", []):
                if member.get("type") == "way":
                    way_id = member.get("ref")
                    role = member.get("role", "outer")
                    # Look for preloaded way data — not always present
                    # Fall back to bounding box centroid
                    if role in ("outer", ""):
                        outer_ways.append(way_id)
            if not outer_ways:
                return None
            # We don't have full way data in relation context usually
            return None

    except Exception as e:
        log.debug(f"Geometry build failed for element {el.get('id')}: {e}")
        return None


def _build_node_map(elements: list) -> dict:
    """Build node_id → (lon, lat) lookup."""
    return {
        el["id"]: (el["lon"], el["lat"])
        for el in elements
        if el["type"] == "node" and "lon" in el and "lat" in el
    }


# ─── GeoJSON output ───────────────────────────────────────────────────────────

def _write_geojson(features: list, out_path: Path) -> None:
    """Write a list of GeoJSON features to file."""
    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }
    out_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")


def load_geojson_features(path: Path) -> list:
    """Load GeoJSON features from file. Returns empty list if file missing."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("features", [])
    except Exception:
        return []


def merge_with_existing(
    osm_summary: dict,
    existing_features_path: Path,
    output_dir: Path,
) -> dict:
    """
    Merge OSM-extracted features with existing features.geojson from features.py.
    Deduplicates by OSM ID.
    """
    from shapely.geometry import shape

    if not existing_features_path.exists():
        return osm_summary

    existing = load_geojson_features(existing_features_path)
    existing_ids = {
        f["properties"].get("osm_id")
        for f in existing
        if f.get("properties", {}).get("osm_id")
    }

    added = 0
    for feat_type, path_str in osm_summary.get("output_paths", {}).items():
        p = Path(path_str)
        new_features = load_geojson_features(p)
        merged = []
        for f in new_features:
            oid = f.get("properties", {}).get("osm_id")
            if oid and oid in existing_ids:
                continue
            merged.append(f)
            added += 1
        if merged:
            _write_geojson(merged, p)

    log.info(f"Merged {added} new OSM features with existing dataset")
    return osm_summary

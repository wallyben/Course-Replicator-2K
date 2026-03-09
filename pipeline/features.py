"""
features.py — Golf feature extraction from OpenStreetMap.

Queries Overpass API for all golf-tagged features within the course boundary,
classifies them, scores confidence, and returns structured GeoJSON + metadata.

Outputs:
  - features.geojson   All features, classified and scored
  - feature_map.png    Colour-coded overhead map
  - holes_metadata.json  Per-hole routing, yardage, par
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
from shapely.geometry import (
    Point, LineString, Polygon, MultiPolygon,
    shape, mapping
)
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer

import importlib.util as _ilu
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
if "config" not in sys.modules or not hasattr(sys.modules["config"], "OVERPASS_URL"):
    _spec = _ilu.spec_from_file_location("config", _CONFIG_PATH)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["config"] = _mod
import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.boundary import _overpass_request

log = logging.getLogger(__name__)


# ─── OSM golf tags ────────────────────────────────────────────────────────────

GOLF_TAGS = {
    "green":        "golf=green",
    "fairway":      "golf=fairway",
    "bunker":       "golf=bunker",
    "tee":          "golf=tee",
    "water":        "golf=water_hazard",
    "rough":        "golf=rough",
    "path":         "golf=path",
    "hole":         "golf=hole",   # Relation type — ordered hole routing
}

FEATURE_COLOURS = {
    "green":   "#2e7d32",   # dark green
    "fairway": "#66bb6a",   # light green
    "bunker":  "#f5deb3",   # wheat/sand
    "tee":     "#1565c0",   # blue
    "water":   "#1e88e5",   # blue
    "rough":   "#827717",   # olive
    "path":    "#795548",   # brown
    "other":   "#9e9e9e",   # grey
}


# ─── Main entry point ─────────────────────────────────────────────────────────

def extract_features(boundary_data: dict, output_dir: Path) -> dict:
    """
    Extract all golf features from OSM within the course boundary.

    Args:
        boundary_data: Output from boundary.resolve_boundary()
        output_dir:    Directory for output files

    Returns:
        {
            "features_geojson": str (path),
            "feature_map_path": str (path),
            "holes": [...],
            "hole_count": int,
            "feature_counts": {...},
            "confidence_summary": {...},
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox_wgs84 = boundary_data["bbox_wgs84"]   # [min_lon, min_lat, max_lon, max_lat]
    course_poly = shape(boundary_data["boundary_wgs84"])

    log.info("Fetching OSM golf features...")
    raw = _fetch_golf_features(bbox_wgs84)
    elements = raw.get("elements", [])

    if not elements:
        log.warning("No OSM golf features found. Feature map will be empty.")

    # Build node lookup
    node_map = {el["id"]: (el["lon"], el["lat"]) for el in elements if el["type"] == "node"}

    # Reconstruct feature geometries
    features = []
    for el in elements:
        if el["type"] not in ("way", "relation"):
            continue
        tags = el.get("tags", {})
        feature_type = _classify_element(tags)
        if feature_type is None:
            continue

        geom = _reconstruct_geometry(el, elements, node_map)
        if geom is None:
            continue

        # Clip to course boundary + small buffer
        clipped = geom.intersection(course_poly.buffer(50))
        if clipped.is_empty:
            continue

        hole_ref = tags.get("ref", tags.get("hole", None))
        par      = _safe_int(tags.get("par"))
        handicap = _safe_int(tags.get("handicap"))

        confidence = _score_confidence(feature_type, clipped, tags, elements)

        features.append({
            "type":        "Feature",
            "geometry":    mapping(clipped),
            "properties": {
                "feature_type": feature_type,
                "osm_id":       el["id"],
                "osm_type":     el["type"],
                "name":         tags.get("name", ""),
                "hole_ref":     hole_ref,
                "par":          par,
                "handicap":     handicap,
                "confidence":   confidence,
                "confidence_label": _confidence_label(confidence),
                "tags":         tags,
            },
        })

    log.info(f"Extracted {len(features)} features from OSM")

    # Count by type
    feature_counts = {}
    for f in features:
        ft = f["properties"]["feature_type"]
        feature_counts[ft] = feature_counts.get(ft, 0) + 1

    # Derive hole routing
    holes = _derive_hole_routing(features, elements, node_map, course_poly)
    log.info(f"Derived {len(holes)} hole routings")

    # Compute confidence summary
    confidence_summary = _summarise_confidence(features, holes)

    # Write GeoJSON
    geojson = {"type": "FeatureCollection", "features": features}
    geojson_path = output_dir / "features.geojson"
    geojson_path.write_text(json.dumps(geojson, indent=2), encoding="utf-8")

    # Write holes metadata — use a distinct file name so routing.py's
    # holes_metadata.json (dict format) is never overwritten by this list.
    osm_holes_path = output_dir / "osm_holes_metadata.json"
    osm_holes_path.write_text(json.dumps(holes, indent=2), encoding="utf-8")

    # Render feature map
    feature_map_path = output_dir / "feature_map.png"
    _render_feature_map(features, boundary_data, feature_map_path)

    # Render per-hole maps
    holes_dir = output_dir / "holes"
    holes_dir.mkdir(exist_ok=True)
    for hole in holes:
        _render_hole_map(hole, features, boundary_data, holes_dir)

    result = {
        "features_geojson":  str(geojson_path),
        "holes_metadata":    str(osm_holes_path),
        "feature_map_path":  str(feature_map_path),
        "holes":             holes,
        "hole_count":        len(holes),
        "feature_counts":    feature_counts,
        "confidence_summary": confidence_summary,
    }

    return result


# ─── OSM fetch ────────────────────────────────────────────────────────────────

def _fetch_golf_features(bbox_wgs84: list) -> dict:
    """Fetch all golf features within bounding box from Overpass."""
    min_lon, min_lat, max_lon, max_lat = bbox_wgs84
    south, west, north, east = min_lat, min_lon, max_lat, max_lon

    query = f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
(
  way["golf"~"green|fairway|bunker|tee|water_hazard|rough|path|hole"]
    ({south},{west},{north},{east});
  relation["golf"~"hole|course"]
    ({south},{west},{north},{east});
  way["natural"="water"]
    ({south},{west},{north},{east});
  way["water"~"pond|lake|river"]
    ({south},{west},{north},{east});
);
out body;
>;
out skel qt;
"""
    return _overpass_request(query)


# ─── Classification ───────────────────────────────────────────────────────────

def _classify_element(tags: dict) -> Optional[str]:
    """Map OSM tags to our feature type vocabulary."""
    golf = tags.get("golf", "")
    if golf in ("green", "fairway", "bunker", "tee", "rough", "path"):
        return golf
    if golf == "water_hazard":
        return "water"
    if golf == "hole":
        return "hole"
    # Natural water within course boundary
    if tags.get("natural") == "water" or tags.get("water") in ("pond", "lake"):
        return "water"
    return None


# ─── Geometry reconstruction ─────────────────────────────────────────────────

def _reconstruct_geometry(el: dict, all_elements: list, node_map: dict):
    """Reconstruct a Shapely geometry from an OSM element."""
    if el["type"] == "way":
        coords = [node_map[n] for n in el.get("nodes", []) if n in node_map]
        if len(coords) < 3:
            return None
        try:
            poly = Polygon(coords)
            return poly if poly.is_valid else poly.buffer(0)
        except Exception:
            return None

    if el["type"] == "relation":
        # Build outer ring from member ways
        way_map = {e["id"]: e for e in all_elements if e["type"] == "way"}
        outer_rings = []
        for member in el.get("members", []):
            if member.get("type") == "way" and member.get("role") in ("outer", ""):
                way = way_map.get(member["ref"])
                if way:
                    coords = [node_map[n] for n in way.get("nodes", []) if n in node_map]
                    if len(coords) >= 3:
                        outer_rings.append(coords)
        if not outer_rings:
            return None
        try:
            polys = [Polygon(r) for r in outer_rings]
            merged = unary_union(polys)
            return merged if merged.is_valid else merged.buffer(0)
        except Exception:
            return None

    return None


# ─── Confidence scoring ───────────────────────────────────────────────────────

def _score_confidence(
    feature_type: str, geom, tags: dict, all_elements: list
) -> float:
    """Return confidence score 0.0–1.0 for a feature."""
    score = 0.5  # baseline

    # Area check
    area_range = config.FEATURE_AREA_RANGES.get(feature_type)
    if area_range and geom.geom_type in ("Polygon", "MultiPolygon"):
        # Project to ITM for metric area
        t = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
        geom_itm = shp_transform(t.transform, geom)
        area_m2 = geom_itm.area
        lo, hi = area_range
        if lo <= area_m2 <= hi:
            score += 0.25
        elif area_m2 < lo * 0.1 or area_m2 > hi * 10:
            score -= 0.30
        else:
            score += 0.05

    # Hole ref present
    if tags.get("ref") or tags.get("hole"):
        score += 0.15

    # Par tag present (greens/fairways)
    if feature_type in ("green", "fairway") and tags.get("par"):
        score += 0.10

    # Geometry validity
    if not geom.is_valid:
        score -= 0.20
    if geom.geom_type == "MultiPolygon":
        score -= 0.05  # fragmented

    return round(max(0.0, min(1.0, score)), 2)


def _confidence_label(score: float) -> str:
    if score >= config.CONFIDENCE_HIGH:
        return "HIGH"
    if score >= config.CONFIDENCE_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _summarise_confidence(features: list, holes: list) -> dict:
    """Produce a per-type confidence summary."""
    by_type: Dict[str, list] = {}
    for f in features:
        ft = f["properties"]["feature_type"]
        cs = f["properties"]["confidence"]
        by_type.setdefault(ft, []).append(cs)

    summary = {}
    for ft, scores in by_type.items():
        avg = sum(scores) / len(scores)
        summary[ft] = {
            "count":   len(scores),
            "avg":     round(avg, 2),
            "label":   _confidence_label(avg),
        }

    # Hole routing confidence
    routed = sum(1 for h in holes if h.get("routing_source") == "osm_relation")
    summary["hole_routing"] = {
        "count":            len(holes),
        "osm_routed":       routed,
        "inferred_routed":  len(holes) - routed,
        "label":            "HIGH" if routed >= len(holes) * 0.7 else "MEDIUM" if routed > 0 else "LOW",
    }
    return summary


# ─── Hole routing derivation ─────────────────────────────────────────────────

def _derive_hole_routing(
    features: list, all_elements: list, node_map: dict, course_poly
) -> List[dict]:
    """
    Derive ordered hole routing from OSM golf=hole relations or by inference.

    Each hole gets:
    {
        "hole_number": int,
        "par": int or null,
        "handicap": int or null,
        "tee_point": [lon, lat],
        "green_point": [lon, lat],
        "green_centroid": [lon, lat],
        "tee_centroid": [lon, lat],
        "length_m": float,
        "length_yards": float,
        "routing_source": "osm_relation" | "inferred",
        "fairway_ids": [osm_id, ...],
        "bunker_ids": [osm_id, ...],
    }
    """
    holes = []

    # First, try OSM golf=hole relations
    hole_relations = [
        el for el in all_elements
        if el["type"] == "relation" and el.get("tags", {}).get("golf") == "hole"
    ]

    for rel in hole_relations:
        tags = rel.get("tags", {})
        hole_num = _safe_int(tags.get("ref", tags.get("hole")))
        if hole_num is None:
            continue

        # Find tee and green centroids from relation members
        tee_centroid   = _find_member_centroid(rel, "tee",   features, node_map, all_elements)
        green_centroid = _find_member_centroid(rel, "green", features, node_map, all_elements)

        if tee_centroid is None or green_centroid is None:
            continue

        length_m     = _haversine(tee_centroid, green_centroid)
        length_yards = length_m * 1.09361

        holes.append({
            "hole_number":    hole_num,
            "par":            _safe_int(tags.get("par")),
            "handicap":       _safe_int(tags.get("handicap")),
            "tee_centroid":   list(tee_centroid),
            "green_centroid": list(green_centroid),
            "length_m":       round(length_m, 1),
            "length_yards":   round(length_yards, 0),
            "routing_source": "osm_relation",
            "osm_id":         rel["id"],
        })

    # If OSM relations are insufficient, infer from tee/green feature pairs
    if len(holes) < 9:
        log.info("Insufficient OSM hole relations — inferring hole routing from tee/green features")
        inferred = _infer_routing(features)
        # Merge: only add inferred holes not already resolved
        existing_nums = {h["hole_number"] for h in holes}
        for h in inferred:
            if h["hole_number"] not in existing_nums:
                holes.append(h)

    # Sort by hole number
    holes.sort(key=lambda h: h["hole_number"])

    # Attach nearby features (bunkers, fairways per hole)
    holes = _attach_nearby_features(holes, features)

    return holes


def _find_member_centroid(
    relation: dict, role: str,
    features: list, node_map: dict, all_elements: list
) -> Optional[Tuple[float, float]]:
    """Find the centroid of a relation member with the given role."""
    for member in relation.get("members", []):
        if member.get("role") != role:
            continue
        # Look in features list
        for f in features:
            if f["properties"]["osm_id"] == member.get("ref"):
                geom = shape(f["geometry"])
                c = geom.centroid
                return (c.x, c.y)
    return None


def _infer_routing(features: list) -> List[dict]:
    """
    Infer hole routing by matching tees to nearest greens.
    Uses a greedy nearest-neighbour assignment.
    """
    tees   = [f for f in features if f["properties"]["feature_type"] == "tee"]
    greens = [f for f in features if f["properties"]["feature_type"] == "green"]

    if not tees or not greens:
        return []

    # Prefer features with hole refs
    def get_ref(f):
        return _safe_int(f["properties"].get("hole_ref"))

    # Group tees by hole ref (take innermost/championship tee = largest area)
    tee_by_ref: dict = {}
    for t in tees:
        ref = get_ref(t)
        if ref:
            geom = shape(t["geometry"])
            t_with_geom = (geom.centroid.x, geom.centroid.y, geom.area)
            if ref not in tee_by_ref or geom.area > tee_by_ref[ref][2]:
                tee_by_ref[ref] = t_with_geom
        else:
            tee_by_ref[f"unknown_{id(t)}"] = (
                shape(t["geometry"]).centroid.x,
                shape(t["geometry"]).centroid.y,
                shape(t["geometry"]).area,
            )

    green_by_ref: dict = {}
    for g in greens:
        ref = get_ref(g)
        geom = shape(g["geometry"])
        centroid = (geom.centroid.x, geom.centroid.y)
        if ref:
            green_by_ref[ref] = centroid
        else:
            green_by_ref[f"unknown_{id(g)}"] = centroid

    holes = []
    assigned_greens = set()

    for ref, tee_data in sorted(tee_by_ref.items(), key=lambda x: x[0] if isinstance(x[0], int) else 99):
        tee_pt = (tee_data[0], tee_data[1])

        # If green with same ref exists
        if ref in green_by_ref:
            green_pt = green_by_ref[ref]
            assigned_greens.add(ref)
        else:
            # Find nearest unassigned green
            nearest = None
            nearest_dist = float("inf")
            for gref, gpt in green_by_ref.items():
                if gref in assigned_greens:
                    continue
                dist = _haversine(tee_pt, gpt)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = (gref, gpt)
            if nearest is None:
                continue
            green_pt = nearest[1]
            assigned_greens.add(nearest[0])

        hole_num = ref if isinstance(ref, int) else len(holes) + 1
        length_m = _haversine(tee_pt, green_pt)
        holes.append({
            "hole_number":    hole_num,
            "par":            None,
            "handicap":       None,
            "tee_centroid":   list(tee_pt),
            "green_centroid": list(green_pt),
            "length_m":       round(length_m, 1),
            "length_yards":   round(length_m * 1.09361, 0),
            "routing_source": "inferred",
            "osm_id":         None,
        })

    return holes


def _attach_nearby_features(holes: list, features: list) -> list:
    """Attach lists of nearby bunker and fairway OSM IDs to each hole."""
    bunkers  = [f for f in features if f["properties"]["feature_type"] == "bunker"]
    fairways = [f for f in features if f["properties"]["feature_type"] == "fairway"]

    for hole in holes:
        tee   = Point(hole["tee_centroid"])
        green = Point(hole["green_centroid"])
        corridor = tee.buffer(0.003).union(green.buffer(0.003))  # rough corridor in WGS84 degrees

        nearby_bunkers  = []
        nearby_fairways = []

        for b in bunkers:
            geom = shape(b["geometry"])
            if corridor.intersects(geom) or geom.distance(tee) < 0.005:
                nearby_bunkers.append({
                    "osm_id":     b["properties"]["osm_id"],
                    "confidence": b["properties"]["confidence"],
                    "centroid":   [geom.centroid.x, geom.centroid.y],
                    "area_m2":    round(_approx_area_m2(geom), 1),
                })

        for fw in fairways:
            geom = shape(fw["geometry"])
            if corridor.intersects(geom):
                nearby_fairways.append({
                    "osm_id":     fw["properties"]["osm_id"],
                    "confidence": fw["properties"]["confidence"],
                })

        hole["bunkers"]  = nearby_bunkers
        hole["fairways"] = nearby_fairways

    return holes


# ─── Map rendering ────────────────────────────────────────────────────────────

def _render_feature_map(features: list, boundary_data: dict, out_path: Path) -> None:
    """Render colour-coded overhead feature map."""
    fig, ax = plt.subplots(figsize=(12, 12), dpi=config.FEATURE_MAP_DPI)

    # Draw boundary
    poly = shape(boundary_data["boundary_wgs84"])
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=0.1, color="#388e3c")
    ax.plot(x, y, color="#1b5e20", linewidth=1.5)

    # Draw features by type (render order: rough → fairway → green → bunker → tee → water)
    render_order = ["rough", "fairway", "green", "bunker", "tee", "water", "path"]
    features_by_type: Dict[str, list] = {}
    for f in features:
        ft = f["properties"]["feature_type"]
        features_by_type.setdefault(ft, []).append(f)

    for ft in render_order:
        colour = FEATURE_COLOURS.get(ft, FEATURE_COLOURS["other"])
        alpha  = 0.6 if ft not in ("path",) else 0.9
        for f in features_by_type.get(ft, []):
            geom = shape(f["geometry"])
            _plot_geom(ax, geom, colour, alpha)

    # Draw hole routing arrows
    ax.set_aspect("equal")
    ax.set_axis_off()

    # Legend
    patches = [
        mpatches.Patch(color=c, label=ft.capitalize())
        for ft, c in FEATURE_COLOURS.items()
        if ft != "other" and ft in features_by_type
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=9)

    centre = boundary_data["centre_wgs84"]
    ax.set_title(
        f"{boundary_data['matched_name']} — Feature Map\n"
        f"Centre: {centre[1]:.4f}°N {centre[0]:.4f}°W",
        pad=10,
    )

    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=config.FEATURE_MAP_DPI)
    plt.close()
    log.info(f"Feature map saved: {out_path}")


def _render_hole_map(hole: dict, features: list, boundary_data: dict, out_dir: Path) -> None:
    """Render a per-hole overhead map."""
    hole_num = hole["hole_number"]
    tee_pt   = Point(hole["tee_centroid"])
    green_pt = Point(hole["green_centroid"])

    # Determine map extent (±200m around the tee–green corridor)
    pad = 0.004  # degrees (~350m)
    minx = min(tee_pt.x, green_pt.x) - pad
    maxx = max(tee_pt.x, green_pt.x) + pad
    miny = min(tee_pt.y, green_pt.y) - pad
    maxy = max(tee_pt.y, green_pt.y) + pad

    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)

    # Draw course boundary
    poly = shape(boundary_data["boundary_wgs84"])
    x, y = poly.exterior.xy
    ax.fill(x, y, alpha=0.08, color="#388e3c")
    ax.plot(x, y, color="#1b5e20", linewidth=0.8)

    # Draw features within extent
    for f in features:
        geom = shape(f["geometry"])
        if not (minx <= geom.centroid.x <= maxx and miny <= geom.centroid.y <= maxy):
            continue
        ft = f["properties"]["feature_type"]
        colour = FEATURE_COLOURS.get(ft, FEATURE_COLOURS["other"])
        _plot_geom(ax, geom, colour, 0.7)

    # Draw routing arrow
    ax.annotate(
        "", xy=(green_pt.x, green_pt.y), xytext=(tee_pt.x, tee_pt.y),
        arrowprops=dict(arrowstyle="->", color="white", lw=2),
    )
    ax.plot(tee_pt.x, tee_pt.y, "bs", markersize=10, label="Tee", zorder=5)
    ax.plot(green_pt.x, green_pt.y, "g^", markersize=10, label="Green", zorder=5)

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.set_axis_off()

    par_str = f"Par {hole['par']}" if hole["par"] else "Par ?"
    yards   = int(hole["length_yards"]) if hole["length_yards"] else "?"
    ax.set_title(f"Hole {hole_num} — {par_str} — {yards} yards\n({hole['routing_source']})", pad=8)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = out_dir / f"hole_{hole_num:02d}_overview.png"
    plt.savefig(str(out_path), bbox_inches="tight", dpi=120)
    plt.close()


def _plot_geom(ax, geom, colour: str, alpha: float) -> None:
    """Plot a Shapely geometry on a matplotlib axis."""
    if geom.is_empty:
        return
    if geom.geom_type == "Polygon":
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=alpha, color=colour)
        ax.plot(x, y, color=colour, linewidth=0.5, alpha=alpha)
    elif geom.geom_type == "MultiPolygon":
        for part in geom.geoms:
            _plot_geom(ax, part, colour, alpha)
    elif geom.geom_type in ("LineString", "LinearRing"):
        x, y = geom.xy
        ax.plot(x, y, color=colour, linewidth=1.5, alpha=alpha)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _haversine(pt1: tuple, pt2: tuple) -> float:
    """Distance in metres between two (lon, lat) points."""
    lon1, lat1 = math.radians(pt1[0]), math.radians(pt1[1])
    lon2, lat2 = math.radians(pt2[0]), math.radians(pt2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def _approx_area_m2(geom) -> float:
    """Approximate area in m² of a WGS84 geometry (rough, for display)."""
    t = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
    geom_itm = shp_transform(t.transform, geom)
    return geom_itm.area


def _safe_int(val) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return None

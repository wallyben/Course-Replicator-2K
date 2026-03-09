"""
boundary.py — Course boundary resolution via multi-source geocoding cascade.

Resolves a golf course name to:
- Boundary polygon (GeoJSON) — the actual OSM course outline, not a bbox
- Bounding box with buffer
- Centre point
- Basic metadata (name, OSM ID, area m²)

Resolution cascade (in order):
  1. Nominatim geocode → precise lat/lon for the named course
  2. Overpass polygon query around geocoded point (radius 500m, then 1500m)
     → finds the actual golf_course polygon regardless of name matching
  3. KNOWN_COURSES bbox centre as geocode seed (when Nominatim fails)
  4. Overpass name-based search (legacy fallback)
  5. KNOWN_COURSES bbox directly (last resort for offline use)
  6. Raise ValueError with helpful --bbox hint

Why Nominatim first?
  Nominatim is a purpose-built geocoder that resolves business names to
  precise coordinates.  An Overpass name-match query for "Old Conna Golf Club"
  will time out on a busy server or return zero results if the OSM name tag
  is spelled differently.  A Nominatim result gives us (lat, lon) which we
  can use to find the actual polygon without relying on name-string matching.

Verification:
  verify_boundary_fit(boundary_data, feature_counts) — called by run_pipeline
  after vision detection to detect wrong-course reconstructions early.
"""

import json
import logging
import math
import random
import time
from typing import List, Optional, Tuple

import requests
from shapely.geometry import shape, box, mapping
from shapely.ops import unary_union
import pyproj
from pyproj import Transformer

import sys
import os
import importlib.util as _ilu

# ── Robust config import ──────────────────────────────────────────────────────
# Always load from the project-root config.py by absolute path so that a
# same-named third-party package (e.g. PyPI "config") cannot shadow it.
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
if "config" not in sys.modules or not hasattr(sys.modules["config"], "OVERPASS_URL"):
    _spec = _ilu.spec_from_file_location("config", _CONFIG_PATH)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["config"] = _mod
import config
# Also ensure the project root is on sys.path for sub-modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

# ─── Overpass endpoint pool ───────────────────────────────────────────────────
# Populated from config; shuffled randomly per request to distribute load and
# work around per-endpoint rate limits (HTTP 429).
# NOTE: getattr default arg is evaluated eagerly by Python, so we compute it
# lazily to avoid AttributeError when a non-project config module is cached.
_OVERPASS_ENDPOINTS: List[str] = getattr(config, "OVERPASS_ENDPOINTS", None) or \
    [getattr(config, "OVERPASS_URL", "https://overpass-api.de/api/interpreter")]

# ─── Area constraints ─────────────────────────────────────────────────────────
_BBOX_AREA_MIN_M2   = 40  * 10_000   # 40 ha  — smallest real 18-hole course
_BBOX_AREA_MAX_M2   = 150 * 10_000   # 150 ha — largest realistic single course

# Overpass search radii for coordinate-based polygon lookup
_RADIUS_TIGHT_M  = 500    # first attempt — should contain the course centre
_RADIUS_WIDE_M   = 1500   # second attempt — if course is large or geocode is offset

# Nominatim endpoint — public instance with required User-Agent header
_NOMINATIM_URL   = getattr(config, "NOMINATIM_URL", "https://nominatim.openstreetmap.org/search")
_NOMINATIM_TIMEOUT = getattr(config, "NOMINATIM_TIMEOUT", 12)

# Verification thresholds
_VERIFY_GREEN_MIN = 8    # fewer than this strongly suggests wrong course
_VERIFY_GREEN_MAX = 35   # more than this is probably multiple courses


# ─── Alias normalisation ─────────────────────────────────────────────────────

def _apply_alias(course_name: str) -> str:
    """
    Look up course_name (lowercase) in config.COURSE_ALIASES.
    Returns the canonical name if found, or the original name unchanged.
    """
    aliases = getattr(config, "COURSE_ALIASES", {})
    return aliases.get(course_name.strip().lower(), course_name)


# ─── Public API ───────────────────────────────────────────────────────────────

def resolve_boundary(course_name: str) -> dict:
    """
    Resolve a golf course name to boundary data.

    Full cascade (PARTS 1–6 of specification):

      0. Alias normalisation   — map colloquial names → canonical OSM name
      1. Hard seed check       — KNOWN_COURSES tight bbox as coordinate seed;
                                 immediately queries Overpass polygon around
                                 that point (does NOT return bbox directly)
      2. Nominatim geocode     — tries multiple name variants until one resolves
      3. Overpass polygon      — around geocoded point, 500m then 1500m radius
      4. Overpass name-search  — legacy string-match fallback
      5. KNOWN_COURSES bbox    — raw rectangle (last resort, no network needed)
      6. Raise ValueError      — with helpful --bbox hint

    Endpoint rotation is applied to every Overpass request (_overpass_request).
    """
    log.info(f"Resolving boundary for: {course_name!r}")

    # ── Step 0: Course alias normalisation ───────────────────────────────────
    canonical = _apply_alias(course_name)
    if canonical != course_name:
        log.info(f"Alias: {course_name!r} → {canonical!r}")
    search_name = canonical   # used for Nominatim + Overpass searches

    # ── Step 1: Hard seed from KNOWN_COURSES (tight coordinate seed) ─────────
    # Look up by original name AND alias in case user typed either form.
    known_bbox = (
        config.KNOWN_COURSES.get(course_name.strip().lower())
        or config.KNOWN_COURSES.get(canonical.strip().lower())
    )
    if known_bbox:
        ctr_lon = (known_bbox[0] + known_bbox[2]) / 2.0
        ctr_lat = (known_bbox[1] + known_bbox[3]) / 2.0
        log.info(
            f"KNOWN_COURSES hard seed: centre ({ctr_lat:.5f}, {ctr_lon:.5f}) "
            f"— querying Overpass polygon (r={_RADIUS_TIGHT_M}m then {_RADIUS_WIDE_M}m)"
        )
        for radius in (_RADIUS_TIGHT_M, _RADIUS_WIDE_M):
            try:
                poly_result = _overpass_polygon_around_point(ctr_lat, ctr_lon, radius)
            except RuntimeError as e:
                log.warning(f"Overpass seed query failed (r={radius}m): {e}")
                poly_result = None
            if poly_result:
                polygon, matched_name, osm_id, osm_type = poly_result
                log.info(
                    f"Hard seed resolved polygon: {matched_name!r} "
                    f"at radius {radius}m"
                )
                return _build_boundary_dict(
                    polygon, course_name, matched_name, osm_id, osm_type,
                    confidence="HIGH",
                )
        # Polygon not found via Overpass — fall back to raw bbox later,
        # but first try Nominatim in case Overpass was rate-limited.
        log.warning(
            f"KNOWN_COURSES Overpass polygon lookup failed — "
            f"will try Nominatim before falling back to raw bbox"
        )

    # ── Step 2: Nominatim geocode (with name variant expansion) ──────────────
    geocode_result = _nominatim_geocode_with_variants(search_name)
    if geocode_result:
        lat, lon, _, _, nom_display = geocode_result
        log.info(f"Nominatim resolved: {nom_display!r} → ({lat:.5f}, {lon:.5f})")

        # ── Step 3: Overpass polygon around geocoded point ───────────────────
        for radius in (_RADIUS_TIGHT_M, _RADIUS_WIDE_M):
            try:
                poly_result = _overpass_polygon_around_point(lat, lon, radius)
            except RuntimeError as e:
                log.warning(f"Overpass polygon query failed (r={radius}m): {e}")
                poly_result = None
            if poly_result:
                polygon, matched_name, osm_id, osm_type = poly_result
                return _build_boundary_dict(
                    polygon, course_name, matched_name, osm_id, osm_type,
                    confidence="HIGH",
                )
        log.warning("Nominatim point found but no Overpass polygon nearby")

    # ── Step 4: Overpass name-search (legacy string-match) ───────────────────
    log.info("Trying Overpass name-search fallback...")
    name_variants = _build_name_variants(search_name)
    elements      = []

    for variant in name_variants:
        log.info(f"Overpass name query: {variant!r}")
        query = _overpass_query_by_name(variant)
        try:
            resp = _overpass_request(query)
        except RuntimeError as e:
            log.warning(f"Overpass name search failed: {e}")
            break
        elements   = resp.get("elements", [])
        candidates = [e for e in elements if e["type"] in ("relation", "way")]
        if candidates:
            log.info(f"Found {len(candidates)} candidate(s) for {variant!r}")
            break
        log.info(f"No OSM results for {variant!r}")

    if elements and [e for e in elements if e["type"] in ("relation", "way")]:
        candidates = [e for e in elements if e["type"] in ("relation", "way")]
        scored = sorted(
            [(
                _name_similarity(
                    search_name.lower(),
                    el.get("tags", {}).get("name", "").lower(),
                ),
                el,
            ) for el in candidates],
            key=lambda x: x[0],
            reverse=True,
        )
        best_score, best_el = scored[0]
        confidence   = "HIGH" if best_score > 0.7 else ("MEDIUM" if best_score > 0.4 else "LOW")
        matched_name = best_el.get("tags", {}).get("name", "Unknown")
        log.info(f"Name-search best match: {matched_name!r} (score={best_score:.2f})")

        polygon = _ways_to_polygon(elements)
        if polygon is None:
            osm_type = best_el["type"]
            osm_id   = best_el["id"]
            try:
                full_resp = _overpass_request(
                    _overpass_query_relation_full(osm_id) if osm_type == "relation"
                    else _overpass_query_way_full(osm_id)
                )
                polygon = _ways_to_polygon(full_resp.get("elements", []))
            except RuntimeError:
                polygon = None

        if polygon is not None:
            return _build_boundary_dict(
                polygon, course_name, matched_name, best_el["id"],
                best_el["type"], confidence=confidence,
            )

    # ── Step 5: KNOWN_COURSES raw bbox (offline last resort) ─────────────────
    if known_bbox:
        log.warning(
            "All network lookups failed — using KNOWN_COURSES raw bbox "
            "(accuracy LOW; delete boundary.json and re-run when network is available)"
        )
        return boundary_from_bbox(*known_bbox, course_name=course_name)

    # ── Step 6: Give up ───────────────────────────────────────────────────────
    _raise_not_found(course_name)


def verify_boundary_fit(
    boundary_data: dict,
    feature_counts: dict,
) -> Tuple[bool, str]:
    """
    Verify the boundary is plausible given the detected feature counts.

    Called by run_pipeline after vision detection.  If the boundary appears
    wrong (wrong course reconstructed), the caller should re-query with a
    larger radius or warn the user.

    Args:
        boundary_data:  boundary dict from resolve_boundary()
        feature_counts: {"green": N, "fairway": N, "bunker": N, ...}

    Returns:
        (is_ok: bool, reason: str)
    """
    green_count = feature_counts.get("green", 0)
    area_ha     = boundary_data.get("area_m2", 0) / 10_000

    if green_count < _VERIFY_GREEN_MIN:
        return (
            False,
            f"Only {green_count} greens detected (expected ≥{_VERIFY_GREEN_MIN}). "
            f"Boundary may cover the wrong course or be too small. "
            f"Area: {area_ha:.1f} ha. Try --bbox for a manual boundary.",
        )

    if green_count > _VERIFY_GREEN_MAX:
        return (
            False,
            f"{green_count} greens detected (expected ≤{_VERIFY_GREEN_MAX}). "
            f"Boundary may cover multiple adjacent courses. "
            f"Area: {area_ha:.1f} ha.",
        )

    if area_ha < 30:
        return (
            False,
            f"Course area {area_ha:.1f} ha is very small for an 18-hole course "
            f"(expected ≥40 ha). Boundary may be incomplete.",
        )

    return (True, f"Boundary OK — {green_count} greens, {area_ha:.1f} ha")


# ─── Nominatim geocoding ──────────────────────────────────────────────────────

def _nominatim_geocode(
    course_name: str,
) -> Optional[Tuple[float, float, Optional[int], Optional[str], str]]:
    """
    Geocode a course name using the Nominatim API.

    Queries Nominatim for the course name, filtering to
    amenity/leisure types (golf_course, leisure centre, etc.).

    Returns (lat, lon, osm_id, osm_type, display_name) or None on failure.
    """
    params = {
        "q":              course_name,
        "format":         "json",
        "limit":          5,
        "addressdetails": 0,
        "extratags":      1,
    }
    headers = {
        "User-Agent": "CourseReplicator2K/3.0 (golf-course-pipeline; research)",
        "Accept-Language": "en",
    }

    try:
        resp = requests.get(
            _NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=_NOMINATIM_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception as e:
        log.warning(f"Nominatim geocode failed: {e}")
        return None

    if not results:
        log.info(f"Nominatim: no results for {course_name!r}")
        return None

    # Prefer results tagged as golf_course/leisure; fall back to first result
    def _is_golf(r: dict) -> bool:
        tags = r.get("extratags", {}) or {}
        return (
            r.get("class") in ("leisure", "amenity", "sport")
            or tags.get("leisure") == "golf_course"
            or "golf" in r.get("display_name", "").lower()
            or "golf" in r.get("type", "").lower()
        )

    golf_results = [r for r in results if _is_golf(r)]
    best = golf_results[0] if golf_results else results[0]

    try:
        lat      = float(best["lat"])
        lon      = float(best["lon"])
        osm_id   = int(best["osm_id"])   if best.get("osm_id")   else None
        osm_type = best.get("osm_type")
        display  = best.get("display_name", course_name)
        return (lat, lon, osm_id, osm_type, display)
    except (KeyError, ValueError, TypeError) as e:
        log.warning(f"Nominatim result parse error: {e}")
        return None


# ─── Nominatim variant expansion ─────────────────────────────────────────────

def _build_nominatim_variants(name: str) -> list:
    """
    Build a list of Nominatim query strings to try for a golf course name.

    Variants (in order):
      1. Original name as-is
      2. "Golf Club"  → "Golf Course"  swap
      3. "Golf Course"→ "Golf Club"    swap
      4. "Golf Club"  → "Golf Links"   swap
      5. Name + " Ireland"
      6. Name + " Bray"
    """
    seen: list = []

    def _add(v: str) -> None:
        if v and v not in seen:
            seen.append(v)

    _add(name)

    # Club ↔ Course / Links substitutions
    for old, new in [
        ("Golf Club",  "Golf Course"),
        ("Golf Course","Golf Club"),
        ("Golf Club",  "Golf Links"),
    ]:
        if old in name:
            _add(name.replace(old, new, 1))

    # Geographic qualifiers
    for qualifier in [" Ireland", " Bray"]:
        _add(name + qualifier)

    return seen


def _nominatim_geocode_with_variants(
    name: str,
) -> Optional[Tuple[float, float, Optional[int], Optional[str], str]]:
    """
    Try _nominatim_geocode() for each variant of name until one resolves.

    Respects Nominatim's 1 req/s rate limit with a 1.1 s sleep between
    unsuccessful attempts (successful first attempt returns immediately).
    """
    variants = _build_nominatim_variants(name)
    for i, variant in enumerate(variants):
        log.info(f"Nominatim geocode attempt {i+1}/{len(variants)}: {variant!r}")
        result = _nominatim_geocode(variant)
        if result:
            return result
        if i < len(variants) - 1:
            time.sleep(1.1)   # 1 req/s Nominatim policy
    return None


# ─── Coordinate-based Overpass polygon query ─────────────────────────────────

def _overpass_polygon_around_point(
    lat: float,
    lon: float,
    radius_m: float,
) -> Optional[Tuple[object, str, int, str]]:
    """
    Query Overpass for a golf_course polygon within radius_m metres of (lat, lon).

    Selects the polygon whose centroid is closest to (lat, lon) and whose area
    is within the valid range [40 ha, 150 ha].

    Returns (polygon, matched_name, osm_id, osm_type) or None.
    """
    query = f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
(
  way["leisure"="golf_course"](around:{radius_m:.0f},{lat},{lon});
  relation["leisure"="golf_course"](around:{radius_m:.0f},{lat},{lon});
  way["landuse"="golf_course"](around:{radius_m:.0f},{lat},{lon});
  relation["landuse"="golf_course"](around:{radius_m:.0f},{lat},{lon});
);
out body;
>;
out skel qt;
"""
    try:
        resp     = _overpass_request(query, retries=3)
        elements = resp.get("elements", [])
    except RuntimeError as e:
        log.warning(f"Overpass around-point query failed: {e}")
        return None

    if not elements:
        return None

    candidates = [e for e in elements if e["type"] in ("relation", "way")]
    if not candidates:
        return None

    log.info(
        f"Overpass around ({lat:.5f},{lon:.5f}) r={radius_m:.0f}m: "
        f"{len(candidates)} candidate(s)"
    )

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    from shapely.ops import transform

    # Reconstruct polygon for each candidate and score by centroid distance + area
    node_map = _reconstruct_nodes(elements)
    best_poly    = None
    best_dist    = float("inf")
    best_name    = "Unknown"
    best_osm_id  = 0
    best_osm_type = "way"

    for el in candidates:
        tags = el.get("tags", {})
        name = tags.get("name", tags.get("alt_name", "Unknown"))

        # Reconstruct polygon for this element
        if el["type"] == "way":
            coords = _reconstruct_way_coords(el.get("nodes", []), node_map)
            poly   = _coords_to_polygon(coords)
        else:
            # Relation: collect member way polygons
            poly = _ways_to_polygon(elements)

        if poly is None:
            continue
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda p: p.area)
        if poly.is_empty or not poly.is_valid:
            poly = poly.buffer(0)

        # Area check
        try:
            poly_itm = transform(t_to_itm.transform, poly)
            area_m2  = poly_itm.area
        except Exception:
            continue

        if area_m2 < _BBOX_AREA_MIN_M2 * 0.5:
            # Allow polygons down to 20 ha (some course outlines are clipped)
            log.debug(f"Candidate {name!r}: too small {area_m2/10000:.1f} ha — skip")
            continue
        if area_m2 > _BBOX_AREA_MAX_M2 * 2.0:
            log.debug(f"Candidate {name!r}: too large {area_m2/10000:.1f} ha — skip")
            continue

        # Distance from query point to polygon centroid
        centroid = poly.centroid
        dist = _haversine(lat, lon, centroid.y, centroid.x)

        if dist < best_dist:
            best_dist     = dist
            best_poly     = poly
            best_name     = name
            best_osm_id   = el["id"]
            best_osm_type = el["type"]

    if best_poly is None:
        return None

    log.info(
        f"Selected polygon: {best_name!r} "
        f"(centroid {best_dist:.0f}m from query point)"
    )
    return (best_poly, best_name, best_osm_id, best_osm_type)


# ─── Boundary dict builder ────────────────────────────────────────────────────

def _build_boundary_dict(
    polygon,
    course_name: str,
    matched_name: str,
    osm_id: int,
    osm_type: str,
    confidence: str = "HIGH",
) -> dict:
    """
    Build the standard boundary dict from a validated Shapely polygon.
    Validates area and adjusts confidence; clips to 40–150 ha via convex hull
    if the polygon is oddly large.
    """
    from shapely.ops import transform

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)

    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    poly_itm  = transform(t_to_itm.transform, polygon)
    area_m2   = poly_itm.area
    area_ha   = area_m2 / 10_000

    log.info(f"Boundary polygon area: {area_ha:.1f} ha — {matched_name!r}")

    # If area outside [30, 200] ha, warn and downgrade confidence
    if not (_BBOX_AREA_MIN_M2 * 0.75 <= area_m2 <= _BBOX_AREA_MAX_M2 * 1.5):
        log.warning(
            f"Boundary area {area_ha:.1f} ha is outside expected range "
            f"[{_BBOX_AREA_MIN_M2/10000:.0f}–{_BBOX_AREA_MAX_M2/10000:.0f} ha]. "
            f"Keeping polygon but confidence reduced."
        )
        confidence = "MEDIUM" if confidence == "HIGH" else confidence

    centroid      = polygon.centroid
    bbox_wgs84    = list(polygon.bounds)
    itm_bounds    = poly_itm.bounds
    buf           = config.BOUNDARY_BUFFER_M
    bbox_buffered = [
        itm_bounds[0] - buf,
        itm_bounds[1] - buf,
        itm_bounds[2] + buf,
        itm_bounds[3] + buf,
    ]

    return {
        "name":              course_name,
        "matched_name":      matched_name,
        "osm_id":            osm_id,
        "osm_type":          osm_type,
        "boundary_wgs84":    mapping(polygon),
        "bbox_wgs84":        bbox_wgs84,
        "bbox_buffered_itm": bbox_buffered,
        "centre_wgs84":      [centroid.x, centroid.y],
        "area_m2":           area_m2,
        "confidence":        confidence,
    }


# ─── Manual fallback ─────────────────────────────────────────────────────────

def boundary_from_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    course_name: str = "Manual Entry"
) -> dict:
    """
    Create boundary data from a manually entered bounding box (WGS84).

    Before accepting the bbox, tries to find the real course polygon by
    querying Overpass around the bbox centre.  Falls back to the bbox
    rectangle if that query fails.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform

    ctr_lat = (min_lat + max_lat) / 2.0
    ctr_lon = (min_lon + max_lon) / 2.0

    # Try to get the real polygon from the bbox centre
    for radius in (_RADIUS_TIGHT_M, _RADIUS_WIDE_M):
        try:
            poly_result = _overpass_polygon_around_point(ctr_lat, ctr_lon, radius)
            if poly_result:
                polygon, matched_name, osm_id, osm_type = poly_result
                log.info(
                    f"boundary_from_bbox: resolved real polygon for "
                    f"{matched_name!r} at radius {radius}m"
                )
                return _build_boundary_dict(
                    polygon, course_name, matched_name, osm_id, osm_type,
                    confidence="MEDIUM",
                )
        except Exception as e:
            log.debug(f"bbox polygon lookup at radius {radius}m failed: {e}")

    # Polygon lookup failed — use refined bbox
    log.info("boundary_from_bbox: using bbox rectangle (polygon lookup failed)")
    refined = _refine_bbox(min_lon, min_lat, max_lon, max_lat, course_name)
    min_lon, min_lat, max_lon, max_lat = refined

    polygon  = shapely_box(min_lon, min_lat, max_lon, max_lat)
    centroid = polygon.centroid

    t_to_itm    = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
    polygon_itm = transform(t_to_itm.transform, polygon)
    itm_bounds  = polygon_itm.bounds
    buf         = config.BOUNDARY_BUFFER_M

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


# ─── Overpass query templates ─────────────────────────────────────────────────

def _overpass_query_by_name(name: str) -> str:
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
    return f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
relation({relation_id});
out body;
>;
out skel qt;
"""


def _overpass_query_way_full(way_id: int) -> str:
    return f"""
[out:json][timeout:{config.OVERPASS_TIMEOUT}];
way({way_id});
out body;
>;
out skel qt;
"""


# ─── Geometry reconstruction ─────────────────────────────────────────────────

def _reconstruct_nodes(elements: list) -> dict:
    return {
        el["id"]: (el["lon"], el["lat"])
        for el in elements
        if el["type"] == "node"
    }


def _reconstruct_way_coords(way_nodes: list, node_map: dict) -> list:
    return [node_map[nid] for nid in way_nodes if nid in node_map]


def _coords_to_polygon(coords: list):
    """Convert a coordinate list to a Shapely Polygon, or None."""
    from shapely.geometry import Polygon
    if len(coords) < 4:
        return None
    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def _ways_to_polygon(elements: list):
    """Reconstruct boundary polygon from Overpass way elements."""
    from shapely.geometry import Polygon, LinearRing
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

    if len(rings) == 1:
        return _coords_to_polygon(rings[0])

    from shapely.geometry import LineString
    lines = [LineString(r) for r in rings]
    polys = list(polygonize(lines))
    if polys:
        return unary_union(polys)
    return None


# ─── FIX 3: Bbox refinement (kept for boundary_from_bbox fallback) ────────────

def _refine_bbox(
    min_lon: float, min_lat: float, max_lon: float, max_lat: float,
    course_name: str,
) -> tuple:
    """
    Refine bounding box to stay within 40–200 ha.
    Uses OSM polygon or fairway cluster sizing.
    """
    from shapely.geometry import box as shapely_box
    from shapely.ops import transform

    t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
    init_area = transform(
        t_to_itm.transform,
        shapely_box(min_lon, min_lat, max_lon, max_lat)
    ).area
    center_lon = (min_lon + max_lon) / 2
    center_lat = (min_lat + max_lat) / 2

    # If already in range, keep
    if _BBOX_AREA_MIN_M2 <= init_area <= _BBOX_AREA_MAX_M2:
        return (min_lon, min_lat, max_lon, max_lat)

    # Try fairway cluster sizing
    try:
        fairway_bbox = _query_fairway_cluster_bbox(min_lon, min_lat, max_lon, max_lat)
        if fairway_bbox is not None:
            return _clamp_bbox_around_centroid(
                *fairway_bbox, center_lon, center_lat, t_to_itm
            )
    except Exception as e:
        log.debug(f"Fairway cluster query failed: {e}")

    return _clamp_bbox_around_centroid(
        min_lon, min_lat, max_lon, max_lat,
        center_lon, center_lat, t_to_itm
    )


def _clamp_bbox_around_centroid(
    min_lon, min_lat, max_lon, max_lat,
    center_lon, center_lat, t_to_itm
) -> tuple:
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

    scale    = math.sqrt(target_area / max(area, 1.0))
    half_lon = max((max_lon - min_lon) / 2 * scale, 0.005)
    half_lat = max((max_lat - min_lat) / 2 * scale, 0.004)

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
    """Query Overpass for a golf_course polygon within the bbox."""
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
    resp     = _overpass_request(query, retries=2)
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
    """Query Overpass for fairway polygons; return their bounding box."""
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
    resp     = _overpass_request(query, retries=2)
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

    return unary_union(polygons).bounds


# ─── Network ──────────────────────────────────────────────────────────────────

def _overpass_request(query: str, retries: int = 3) -> dict:
    """
    Send an Overpass QL query with endpoint rotation and 429 backoff.

    Shuffles the endpoint pool on every call so load is distributed.
    On HTTP 429 (rate-limit) or connection error the next endpoint in
    the pool is tried after an exponential backoff (2s, 4s, 8s, …).
    """
    endpoints = list(_OVERPASS_ENDPOINTS)
    random.shuffle(endpoints)
    # Build a pool long enough to cover `retries` attempts,
    # cycling through endpoints if retries > pool size.
    pool = (endpoints * ((retries // max(len(endpoints), 1)) + 2))[:retries]

    last_error: Exception = RuntimeError("No attempts made")
    for attempt, endpoint in enumerate(pool):
        try:
            resp = requests.post(
                endpoint,
                data={"data": query},
                timeout=config.OVERPASS_TIMEOUT + 10,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt
                log.warning(
                    f"Overpass 429 rate-limit at {endpoint} "
                    f"(attempt {attempt+1}/{retries}) — waiting {wait}s then trying next endpoint"
                )
                time.sleep(wait)
                last_error = requests.HTTPError(f"429 from {endpoint}", response=resp)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_error = e
            wait = 2 ** attempt
            log.warning(
                f"Overpass request failed (attempt {attempt+1}/{retries}, {endpoint}): {e}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Overpass API unavailable after {retries} attempts: {last_error}"
    ) from last_error


# ─── String helpers ───────────────────────────────────────────────────────────

def _name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    stop     = {"golf", "club", "course", "links", "gc", "the"}
    tokens_a = set(a.split()) - stop
    tokens_b = set(b.split()) - stop
    if not tokens_a or not tokens_b:
        return 0.5
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def _build_name_variants(course_name: str) -> list:
    variants = [course_name]
    suffixes = [
        " Golf Club", " Golf Links", " Golf Course", " Golf & Country Club",
        " Golf", " Club", " Links", " Course",
    ]
    working = course_name
    for suffix in suffixes:
        if working.lower().endswith(suffix.lower()):
            working = working[: len(working) - len(suffix)].strip()
            if working and working not in variants:
                variants.append(working)

    stop  = {"golf", "club", "course", "links", "the", "old", "new", "royal"}
    words = [w for w in working.split() if w.lower() not in stop]
    if words and len(words[-1]) > 3:
        last_word = words[-1]
        if last_word not in variants and last_word != working:
            variants.append(last_word)

    return variants


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = phi2 - phi1
    dlam       = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(min(a, 1.0)))


def _raise_not_found(course_name: str) -> None:
    key = course_name.strip().lower()
    raise ValueError(
        f"No golf course found for: {course_name!r}\n"
        f"Nominatim geocode and all Overpass queries returned no results.\n\n"
        f"Options:\n"
        f"  1. Add to KNOWN_COURSES in config.py:\n"
        f'     "{key}": [min_lon, min_lat, max_lon, max_lat],\n'
        f"  2. Use --bbox on the command line:\n"
        f"     python scripts/run_pipeline.py \"{course_name}\" "
        f"--bbox min_lon,min_lat,max_lon,max_lat\n"
        f"     (Find coordinates at openstreetmap.org — right-click → 'Show address')"
    )

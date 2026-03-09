"""
routing_diagram_ai.py — Extract golf course layout from a routing diagram image.

Inputs
------
• A PNG/JPEG routing diagram (architect's layout, brochure map, course guide)
• boundary_data dict (for pixel → WGS84 coordinate mapping)
• output_dir (pipeline output directory)

Outputs (written to output_dir)
---------------------------------
• routing_fairways.geojson
• routing_greens.geojson
• routing_tees.geojson
• routing_bunkers.geojson
• routing_water.geojson
• routing_holes.geojson   ← tee→green LineStrings with hole metadata

Feature priority in fusion
---------------------------
    routing_diagram  >  satellite vision  >  OSM

Usage
-----
    from pipeline.routing_diagram_ai import extract_routing_diagram, fuse_diagram_features

    result = extract_routing_diagram(
        diagram_path   = "assets/routing_diagrams/old_conna_layout.png",
        boundary_data  = boundary_data,
        output_dir     = output_dir,
    )
    fuse_diagram_features(result, output_dir)

Dependencies
------------
• opencv-python   (cv2)             — already in requirements.txt
• numpy, Pillow, shapely, scipy     — already in requirements.txt
• pytesseract [OPTIONAL]            — pip install pytesseract + tesseract binary
  If not available: hole numbers are skipped gracefully.
"""

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Runtime-optional heavy imports ───────────────────────────────────────────

def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for routing diagram extraction.\n"
            "Install it with:  pip install opencv-python"
        )


def _try_pytesseract():
    """Return pytesseract module or None if not installed."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()   # verify binary is present
        return pytesseract
    except Exception:
        return None


# ── Default HSV colour thresholds ────────────────────────────────────────────
# OpenCV HSV convention: H ∈ [0,180], S ∈ [0,255], V ∈ [0,255]
#
# Multiple ranges are tried in order and OR-merged.
# Calibrated for typical golf brochure / architect routing diagrams.

_COLOUR_RANGES: Dict[str, List[Dict]] = {

    # Medium / bright green corridors
    "fairway": [
        {"h": (30, 88),  "s": (35, 230), "v": (50, 210)},
        {"h": (28, 85),  "s": (20, 160), "v": (30, 160)},   # muted/printed
    ],

    # Greens use the same colour palette; separation is done by shape
    # (circularity) and size, NOT by a distinct hue.
    # A separate _DARKER_GREEN range catches diagrams where greens are
    # a distinctly different shade (darker / lighter).
    "green_distinct": [
        {"h": (35, 90),  "s": (60, 255), "v": (120, 255)},  # bright flag-green
        {"h": (90, 130), "s": (30, 160), "v": (80, 220)},   # blue-green variant
    ],

    # Sand / bunkers
    "bunker": [
        {"h": (12, 45),  "s": (10, 130), "v": (160, 255)},  # cream/sand
        {"h": (10, 42),  "s": (5,  70),  "v": (195, 255)},  # near-white sand
        {"h": (18, 50),  "s": (40, 180), "v": (140, 255)},  # warm yellow
    ],

    # Water bodies
    "water": [
        {"h": (88, 140), "s": (45, 255), "v": (35, 220)},   # blue / cyan
        {"h": (85, 135), "s": (80, 255), "v": (20, 160)},   # dark navy
        {"h": (100, 140),"s": (20, 120), "v": (150, 255)},  # pale blue
    ],

    # Tee markers (coloured rectangles near hole start)
    # Championship (black/dark), medal (yellow), stroke (white), ladies (red)
    "tee_red": [
        {"h": (0,  12),  "s": (130, 255), "v": (120, 255)},
        {"h": (168, 180),"s": (130, 255), "v": (120, 255)},
    ],
    "tee_yellow": [
        {"h": (18, 38),  "s": (100, 255), "v": (140, 255)},
    ],
    "tee_blue": [
        {"h": (95, 130), "s": (90, 255),  "v": (80, 255)},
    ],
    "tee_white": [
        {"h": (0,  180), "s": (0,  55),   "v": (195, 255)},
    ],
    "tee_black": [
        {"h": (0,  180), "s": (0,  60),   "v": (0,  55)},
    ],
}

# ── Area / shape thresholds (pixel-squared) ───────────────────────────────────
# Computed as fraction of total image area — scaled when image is loaded.

_AREA_FRACTIONS = {
    "fairway":    (0.001,  0.25),   # 0.1 % – 25 % of image
    "green":      (0.0002, 0.04),   # 0.02 % – 4 %
    "bunker":     (0.0001, 0.06),   # 0.01 % – 6 %
    "water":      (0.0002, 0.30),   # 0.02 % – 30 %
    "tee":        (0.00005, 0.01),  # 0.005 % – 1 %
}

_GREEN_CIRCULARITY_MIN  = 0.45   # shapes above this are greens (if also small)
_GREEN_MAX_AREA_FRAC    = 0.04   # greens cannot be more than 4 % of image
_FAIRWAY_CIRC_MAX       = 0.70   # fairways are never very circular

# Contour simplification epsilon (fraction of contour arc length)
_SIMPLIFY_EPS_FRAC = 0.006

# ── Geometry helpers ──────────────────────────────────────────────────────────

def _circularity(area: float, perimeter: float) -> float:
    if perimeter < 1e-6:
        return 0.0
    return 4.0 * math.pi * area / (perimeter * perimeter)


def _centroid(contour) -> Tuple[float, float]:
    import cv2
    M = cv2.moments(contour)
    if M["m00"] < 1e-6:
        pts = contour.reshape(-1, 2)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def _contour_arc_length(c) -> float:
    import cv2
    return cv2.arcLength(c, closed=True)


def _pt_dist(a, b) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _segment_point_dist(px: float, py: float,
                        ax: float, ay: float,
                        bx: float, by: float) -> float:
    """Distance from point P to line segment AB."""
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return _pt_dist((px, py), (ax, ay))
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)))
    return _pt_dist((px, py), (ax + t * dx, ay + t * dy))


def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


# ── Coordinate transform ──────────────────────────────────────────────────────

def _px_to_lonlat(
    px: float, py: float,
    img_w: int, img_h: int,
    bbox: List[float],
) -> Tuple[float, float]:
    """
    Map pixel (px, py) to WGS84 (lon, lat).

    The image origin is assumed to be the top-left corner of the course bbox.
    Longitude increases left→right; latitude decreases top→bottom (y-inverted).

    bbox = [min_lon, min_lat, max_lon, max_lat]
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + (px / img_w)  * (max_lon - min_lon)
    lat = max_lat - (py / img_h)  * (max_lat - min_lat)
    return lon, lat


# ── GeoJSON builders ──────────────────────────────────────────────────────────

def _make_feature(geometry: dict, props: dict) -> dict:
    props.setdefault("source", "routing_diagram")
    props.setdefault("id", str(uuid.uuid4()))
    return {"type": "Feature", "geometry": geometry, "properties": props}


def _feature_collection(features: List[dict]) -> dict:
    return {"type": "FeatureCollection", "features": features}


def _write_geojson(features: List[dict], path: Path) -> None:
    path.write_text(
        json.dumps(_feature_collection(features), indent=2),
        encoding="utf-8",
    )
    log.info(f"  Written {len(features):3d} features → {path.name}")


# ── Core extractor class ──────────────────────────────────────────────────────

class RoutingDiagramExtractor:
    """
    Extract golf course features from a routing diagram image.

    Parameters
    ----------
    diagram_path : Path
        Input image (PNG / JPEG).
    boundary_data : dict
        Output of boundary.resolve_boundary(); provides bbox_wgs84.
    output_dir : Path
        Pipeline output directory; GeoJSON files are written here.
    colour_ranges : dict, optional
        Override default HSV colour thresholds.
    """

    def __init__(
        self,
        diagram_path: Path,
        boundary_data: dict,
        output_dir: Path,
        colour_ranges: Optional[dict] = None,
    ) -> None:
        self.diagram_path   = Path(diagram_path)
        self.boundary_data  = boundary_data
        self.output_dir     = Path(output_dir)
        self.bbox           = boundary_data["bbox_wgs84"]
        self.colour_ranges  = colour_ranges or _COLOUR_RANGES

        self._img_rgb: Optional[np.ndarray] = None
        self._img_hsv: Optional[np.ndarray] = None
        self._img_w: int = 0
        self._img_h: int = 0

    # ── Public entry point ────────────────────────────────────────────────────

    def extract(self) -> dict:
        """
        Run full extraction pipeline.

        Returns
        -------
        dict with keys:
            fairways, greens, bunkers, water, tees, holes
            Each value is a list of GeoJSON Feature dicts.
        """
        log.info(f"  Loading diagram: {self.diagram_path.name}")
        self._load_image()

        log.info("  Segmenting features by colour…")
        all_green_features = self._segment_green_areas()
        fairways, greens   = self._classify_greens_vs_fairways(all_green_features)
        bunkers            = self._segment_mask("bunker",  "bunker")
        water              = self._segment_mask("water",   "water")
        tees               = self._detect_tees()

        log.info(
            f"  Raw counts — fairways:{len(fairways)}  greens:{len(greens)}  "
            f"bunkers:{len(bunkers)}  water:{len(water)}  tees:{len(tees)}"
        )

        log.info("  Running OCR for hole numbers…")
        hole_positions = self._detect_hole_numbers()
        log.info(f"  OCR found {len(hole_positions)} hole number(s).")

        log.info("  Associating holes…")
        hole_map = self._associate_holes(hole_positions, tees, greens, fairways)

        log.info("  Constructing routing…")
        holes = self._construct_routing(hole_map, tees, greens, fairways)
        log.info(f"  Routing: {len(holes)} hole route(s).")

        # Stamp hole_number onto matched features
        for hole_num, assigned in hole_map.items():
            for layer, features in [("tee", tees), ("green", greens), ("fairway", fairways)]:
                fid = assigned.get(layer + "_id")
                if fid:
                    for f in features:
                        if f.get("properties", {}).get("id") == fid:
                            f["properties"]["hole_number"] = hole_num

        return {
            "fairways": fairways,
            "greens":   greens,
            "bunkers":  bunkers,
            "water":    water,
            "tees":     tees,
            "holes":    holes,
        }

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_image(self) -> None:
        cv2 = _require_cv2()
        img_bgr = cv2.imread(str(self.diagram_path))
        if img_bgr is None:
            raise ValueError(f"Cannot load image: {self.diagram_path}")

        self._img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        self._img_h, self._img_w = img_bgr.shape[:2]
        log.info(f"  Image size: {self._img_w} × {self._img_h} px")

    # ── Mask helpers ──────────────────────────────────────────────────────────

    def _colour_mask(self, ranges_key: str) -> np.ndarray:
        """
        Build a binary mask from all HSV ranges for a given key.
        Returns uint8 array (255 = inside range, 0 = outside).
        """
        cv2  = _require_cv2()
        hsv  = self._img_hsv
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in self.colour_ranges.get(ranges_key, []):
            lo = np.array([r["h"][0], r["s"][0], r["v"][0]], dtype=np.uint8)
            hi = np.array([r["h"][1], r["s"][1], r["v"][1]], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask

    def _clean_mask(self, mask: np.ndarray, open_k: int = 3, close_k: int = 7) -> np.ndarray:
        """Morphological open (noise removal) + close (fill holes)."""
        cv2 = _require_cv2()
        ko  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k,  open_k))
        kc  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  ko)
        out = cv2.morphologyEx(out,  cv2.MORPH_CLOSE, kc)
        return out

    def _contours_from_mask(
        self,
        mask: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> list:
        """Extract and filter contours by area."""
        cv2 = _require_cv2()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [
            c for c in cnts
            if min_area <= cv2.contourArea(c) <= max_area
        ]

    def _area_bounds(self, key: str) -> Tuple[float, float]:
        frac   = _AREA_FRACTIONS[key]
        total  = self._img_w * self._img_h
        return frac[0] * total, frac[1] * total

    def _contour_to_geojson_feature(
        self,
        contour,
        feature_type: str,
        extra_props: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Convert a pixel-space contour to a WGS84 GeoJSON Polygon feature.
        Returns None if simplification produces fewer than 3 vertices.
        """
        cv2 = _require_cv2()
        arc = _contour_arc_length(contour)
        eps = max(1.5, _SIMPLIFY_EPS_FRAC * arc)
        simplified = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)

        if len(simplified) < 3:
            return None

        coords = [
            _px_to_lonlat(float(x), float(y), self._img_w, self._img_h, self.bbox)
            for x, y in simplified
        ]
        coords.append(coords[0])   # close ring

        # Shapely simplification for smoother output
        try:
            from shapely.geometry import Polygon
            from shapely.errors import TopologicalError
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid or poly.is_empty:
                return None
            # Simplify by ~1 metre in WGS84 degrees
            deg_per_m = 1.0 / 111_000
            poly = poly.simplify(2 * deg_per_m, preserve_topology=True)
            if poly.is_empty:
                return None
            from shapely.geometry import mapping
            geom = mapping(poly)
        except Exception:
            geom = {"type": "Polygon", "coordinates": [coords]}

        # Compute approx area in m²
        cx, cy   = _centroid(contour)
        clon, clat = _px_to_lonlat(cx, cy, self._img_w, self._img_h, self.bbox)
        px_area  = float(cv2.contourArea(contour))
        img_area = self._img_w * self._img_h
        bbox_m2  = (
            _haversine_m(self.bbox[0], (self.bbox[1]+self.bbox[3])/2,
                         self.bbox[2], (self.bbox[1]+self.bbox[3])/2)
            * _haversine_m((self.bbox[0]+self.bbox[2])/2, self.bbox[1],
                           (self.bbox[0]+self.bbox[2])/2, self.bbox[3])
        )
        area_m2 = round(px_area / img_area * bbox_m2, 1)

        fid  = str(uuid.uuid4())
        props = dict(extra_props or {})
        props.update({
            "id":         fid,
            "type":       feature_type,
            "source":     "routing_diagram",
            "area_m2":    area_m2,
            "centroid_px": [round(cx, 1), round(cy, 1)],
        })
        return {"type": "Feature", "geometry": geom, "properties": props}

    # ── Green area segmentation + classification ──────────────────────────────

    def _segment_green_areas(self) -> list:
        """Segment all 'playable green' areas regardless of type."""
        cv2    = _require_cv2()
        # Merge fairway colour ranges (the common green detection)
        mask   = self._colour_mask("fairway")
        # Also include the distinct green range if defined
        mask   = cv2.bitwise_or(mask, self._colour_mask("green_distinct"))
        mask   = self._clean_mask(mask, open_k=5, close_k=11)

        min_a, max_a = self._area_bounds("fairway")   # use fairway bounds (larger)
        contours = self._contours_from_mask(mask, min_a * 0.2, max_a)

        features = []
        for c in contours:
            feat = self._contour_to_geojson_feature(c, "green_area")
            if feat:
                cv2_area = cv2.contourArea(c)
                arc      = _contour_arc_length(c)
                circ     = _circularity(cv2_area, arc)
                feat["properties"]["_circularity"] = round(circ, 3)
                feat["properties"]["_px_area"]     = round(cv2_area, 1)
                features.append(feat)
        return features

    def _classify_greens_vs_fairways(
        self, all_green: list
    ) -> Tuple[List[dict], List[dict]]:
        """
        Split green-coloured areas into fairways vs putting greens using
        circularity + area heuristics.
        """
        total_img_area = self._img_w * self._img_h
        green_max_px   = _GREEN_MAX_AREA_FRAC * total_img_area

        fairways: List[dict] = []
        greens:   List[dict] = []

        for f in all_green:
            px_area  = f["properties"].get("_px_area",     0)
            circ     = f["properties"].get("_circularity", 0)
            is_green = (
                circ   >= _GREEN_CIRCULARITY_MIN and
                px_area <= green_max_px
            )
            # Clean up internal fields
            for k in ("_circularity", "_px_area"):
                f["properties"].pop(k, None)

            if is_green:
                f["properties"]["type"] = "green"
                greens.append(f)
            else:
                f["properties"]["type"] = "fairway"
                fairways.append(f)

        return fairways, greens

    # ── Generic mask → features ───────────────────────────────────────────────

    def _segment_mask(self, colour_key: str, feature_type: str) -> List[dict]:
        """Segment a single feature type using colour key and area bounds."""
        mask     = self._colour_mask(colour_key)
        mask     = self._clean_mask(mask, open_k=3, close_k=7)
        min_a, max_a = self._area_bounds(feature_type if feature_type in _AREA_FRACTIONS else "bunker")
        contours = self._contours_from_mask(mask, min_a, max_a)

        features = []
        for c in contours:
            feat = self._contour_to_geojson_feature(c, feature_type)
            if feat:
                features.append(feat)
        return features

    # ── Tee detection ─────────────────────────────────────────────────────────

    def _detect_tees(self) -> List[dict]:
        """
        Detect tee boxes: small, roughly rectangular coloured shapes.
        Tries several colour ranges and merges results by NMS.
        """
        cv2  = _require_cv2()
        tee_keys = ["tee_red", "tee_yellow", "tee_blue", "tee_white", "tee_black"]
        min_a, max_a = self._area_bounds("tee")

        combined_mask = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        for key in tee_keys:
            m = self._colour_mask(key)
            combined_mask = cv2.bitwise_or(combined_mask, m)

        combined_mask = self._clean_mask(combined_mask, open_k=2, close_k=5)
        contours      = self._contours_from_mask(combined_mask, min_a, max_a)

        features = []
        for c in contours:
            rect  = cv2.minAreaRect(c)
            w, h  = rect[1]
            if w < 1 or h < 1:
                continue
            aspect = max(w, h) / min(w, h)
            if aspect < 1.2 or aspect > 8.0:   # tees are rectangular
                continue
            feat = self._contour_to_geojson_feature(c, "tee")
            if feat:
                features.append(feat)

        # Non-maximum suppression by centroid proximity
        return self._nms_by_centroid(features, min_dist_px=30)

    # ── OCR ───────────────────────────────────────────────────────────────────

    def _detect_hole_numbers(self) -> List[dict]:
        """
        Use pytesseract to find digit labels (1–18) in the diagram.

        Returns list of:
          {"number": int, "x": float, "y": float, "conf": float}
        """
        tess = _try_pytesseract()
        if tess is None:
            log.warning("  pytesseract not available — hole number OCR skipped.")
            log.warning("  Install with:  pip install pytesseract")
            return []

        cv2  = _require_cv2()
        gray = cv2.cvtColor(self._img_rgb, cv2.COLOR_RGB2GRAY)

        # Try multiple preprocessed versions, keep highest-confidence results
        candidates: Dict[int, dict] = {}   # number → best hit

        for scale in [1.0, 1.5, 2.0]:
            if scale != 1.0:
                w = int(self._img_w * scale)
                h = int(self._img_h * scale)
                resized = cv2.resize(gray, (w, h))
            else:
                resized = gray
                w, h = self._img_w, self._img_h

            # OTSU threshold
            _, thresh = cv2.threshold(
                resized, 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            for src_img in [resized, thresh]:
                try:
                    data = tess.image_to_data(
                        src_img,
                        config="--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789",
                        output_type=tess.Output.DICT,
                    )
                except Exception as e:
                    log.debug(f"pytesseract error: {e}")
                    continue

                for i, text in enumerate(data["text"]):
                    text = text.strip()
                    if not text.isdigit():
                        continue
                    num  = int(text)
                    if num < 1 or num > 18:
                        continue
                    try:
                        conf = float(data["conf"][i])
                    except (TypeError, ValueError):
                        continue
                    if conf < 30:
                        continue
                    # Map back to original pixel coords
                    cx = (data["left"][i] + data["width"][i]  / 2) / scale
                    cy = (data["top"][i]  + data["height"][i] / 2) / scale
                    if num not in candidates or conf > candidates[num]["conf"]:
                        candidates[num] = {
                            "number": num,
                            "x": cx, "y": cy,
                            "conf": conf,
                        }

        results = sorted(candidates.values(), key=lambda d: d["number"])
        return results

    # ── Hole association ──────────────────────────────────────────────────────

    def _feature_centroid_px(self, feature: dict) -> Tuple[float, float]:
        """Return pixel centroid from stored _centroid_px or geometry."""
        p = feature.get("properties", {})
        cp = p.get("centroid_px")
        if cp:
            return float(cp[0]), float(cp[1])
        # fallback: centroid of first polygon ring in pixel space
        geom   = feature.get("geometry", {})
        coords = geom.get("coordinates", [[]])[0]
        if not coords:
            return 0.0, 0.0
        # Convert WGS84 back to approximate pixel (inverse of _px_to_lonlat)
        min_lon, min_lat, max_lon, max_lat = self.bbox
        px_list = [
            ((lon - min_lon) / (max_lon - min_lon) * self._img_w,
             (max_lat - lat) / (max_lat - min_lat) * self._img_h)
            for lon, lat in coords
        ]
        xs = [pt[0] for pt in px_list]
        ys = [pt[1] for pt in px_list]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _associate_holes(
        self,
        hole_positions: list,
        tees: list,
        greens: list,
        fairways: list,
    ) -> dict:
        """
        Match OCR-detected hole numbers to the nearest tee, green, fairway.

        Returns dict: {hole_num: {tee_id, green_id, fairway_id, tee_px, green_px}}
        """
        hole_map: dict = {}

        # Precompute pixel centroids for each feature set
        def pxc(features):
            return [(self._feature_centroid_px(f), f["properties"]["id"]) for f in features]

        tee_pxcs     = pxc(tees)
        green_pxcs   = pxc(greens)
        fairway_pxcs = pxc(fairways)

        max_tee_dist     = 0.20 * max(self._img_w, self._img_h)
        max_green_dist   = 0.45 * max(self._img_w, self._img_h)
        max_fairway_dist = 0.50 * max(self._img_w, self._img_h)

        used_tees    = set()
        used_greens  = set()
        used_fairways = set()

        for hp in hole_positions:
            hx, hy = hp["x"], hp["y"]
            num    = hp["number"]

            # Nearest unassigned tee
            best_t = None
            best_t_d = float("inf")
            for (cx, cy), fid in tee_pxcs:
                if fid in used_tees:
                    continue
                d = _pt_dist((hx, hy), (cx, cy))
                if d < best_t_d and d < max_tee_dist:
                    best_t_d, best_t = d, fid

            # Nearest unassigned green — may be farther from hole number
            best_g = None
            best_g_d = float("inf")
            for (cx, cy), fid in green_pxcs:
                if fid in used_greens:
                    continue
                d = _pt_dist((hx, hy), (cx, cy))
                if d < best_g_d and d < max_green_dist:
                    best_g_d, best_g = d, fid

            # Nearest unassigned fairway whose centroid lies between tee and green
            tee_px = next(
                (c for c, fid in tee_pxcs if fid == best_t),
                (hx, hy),
            )
            grn_px = next(
                (c for c, fid in green_pxcs if fid == best_g),
                (hx, hy),
            )
            best_f = None
            best_f_d = float("inf")
            for (cx, cy), fid in fairway_pxcs:
                if fid in used_fairways:
                    continue
                d = _segment_point_dist(cx, cy,
                                        tee_px[0], tee_px[1],
                                        grn_px[0], grn_px[1])
                if d < best_f_d and d < max_fairway_dist:
                    best_f_d, best_f = d, fid

            hole_map[num] = {
                "tee_id":      best_t,
                "green_id":    best_g,
                "fairway_id":  best_f,
                "tee_px":      tee_px,
                "green_px":    grn_px,
                "number_px":   (hx, hy),
            }
            if best_t:     used_tees.add(best_t)
            if best_g:     used_greens.add(best_g)
            if best_f:     used_fairways.add(best_f)

        # For unassigned greens / tees, try to infer hole number from spatial order
        if len(hole_map) < 9:
            hole_map = self._infer_remaining_holes(
                hole_map, tees, greens, fairways,
                used_tees, used_greens, used_fairways,
            )

        return hole_map

    def _infer_remaining_holes(
        self, hole_map, tees, greens, fairways,
        used_tees, used_greens, used_fairways,
    ) -> dict:
        """
        For courses where OCR found fewer than 9 hole numbers, try to pair
        remaining tees and greens spatially by nearest-neighbour.
        """
        unassigned_tees   = [f for f in tees   if f["properties"]["id"] not in used_tees]
        unassigned_greens = [f for f in greens  if f["properties"]["id"] not in used_greens]
        unassigned_fways  = [f for f in fairways if f["properties"]["id"] not in used_fairways]

        # Use next available hole number
        next_num = max(hole_map.keys(), default=0) + 1

        # Pair tees to nearest green
        for tee in unassigned_tees:
            if next_num > 18:
                break
            tpx = self._feature_centroid_px(tee)
            best_g, best_d = None, float("inf")
            for g in unassigned_greens:
                gpx = self._feature_centroid_px(g)
                d   = _pt_dist(tpx, gpx)
                if d < best_d:
                    best_d, best_g = d, g

            # Nearest fairway along tee→green
            if best_g:
                gpx = self._feature_centroid_px(best_g)
                best_f, best_fd = None, float("inf")
                for fw in unassigned_fways:
                    fwpx = self._feature_centroid_px(fw)
                    d    = _segment_point_dist(
                        fwpx[0], fwpx[1],
                        tpx[0], tpx[1], gpx[0], gpx[1],
                    )
                    if d < best_fd:
                        best_fd, best_f = d, fw

                hole_map[next_num] = {
                    "tee_id":    tee["properties"]["id"],
                    "green_id":  best_g["properties"]["id"],
                    "fairway_id": best_f["properties"]["id"] if best_f else None,
                    "tee_px":   tpx,
                    "green_px": gpx,
                    "number_px": tpx,
                }
                used_tees.add(tee["properties"]["id"])
                used_greens.add(best_g["properties"]["id"])
                if best_f:
                    used_fairways.add(best_f["properties"]["id"])
                unassigned_greens = [g for g in unassigned_greens
                                     if g["properties"]["id"] not in used_greens]
                unassigned_fways  = [f for f in unassigned_fways
                                     if f["properties"]["id"] not in used_fairways]
                next_num += 1

        return hole_map

    # ── Routing construction ──────────────────────────────────────────────────

    def _construct_routing(
        self,
        hole_map: dict,
        tees: list,
        greens: list,
        fairways: list,
    ) -> List[dict]:
        """Build a tee → [fairway midpoint] → green LineString for each hole."""
        def find_by_id(features, fid):
            return next((f for f in features
                         if f["properties"]["id"] == fid), None)

        def geom_centroid_wgs84(feature):
            """Return (lon, lat) centroid from WGS84 geometry."""
            geom = feature.get("geometry", {})
            coords = geom.get("coordinates", [[]])[0]
            if not coords:
                return None
            lons = [p[0] for p in coords]
            lats = [p[1] for p in coords]
            return sum(lons) / len(lons), sum(lats) / len(lats)

        holes: List[dict] = []

        for hole_num in sorted(hole_map.keys()):
            asgn  = hole_map[hole_num]
            tee   = find_by_id(tees,     asgn.get("tee_id"))
            green = find_by_id(greens,   asgn.get("green_id"))
            fw    = find_by_id(fairways, asgn.get("fairway_id"))

            if not tee and not green:
                continue

            route_coords: List[List[float]] = []

            if tee:
                tc = geom_centroid_wgs84(tee)
                if tc:
                    route_coords.append(list(tc))
            if fw:
                fc = geom_centroid_wgs84(fw)
                if fc:
                    route_coords.append(list(fc))
            if green:
                gc = geom_centroid_wgs84(green)
                if gc:
                    route_coords.append(list(gc))

            # De-duplicate consecutive identical points
            deduped: List[List[float]] = []
            for pt in route_coords:
                if not deduped or pt != deduped[-1]:
                    deduped.append(pt)

            if len(deduped) < 2:
                continue

            # Compute distance
            dist_m = sum(
                _haversine_m(deduped[i][0], deduped[i][1],
                             deduped[i+1][0], deduped[i+1][1])
                for i in range(len(deduped) - 1)
            )
            dist_yd = round(dist_m * 1.09361)

            # Infer par
            if dist_yd < 250:
                par = 3
            elif dist_yd > 470:
                par = 5
            else:
                par = 4

            hole_feat = _make_feature(
                {"type": "LineString", "coordinates": deduped},
                {
                    "hole_number":    hole_num,
                    "par":            par,
                    "distance_m":     round(dist_m, 1),
                    "distance_yards": dist_yd,
                    "length_m":       round(dist_m, 1),
                    "length_yards":   dist_yd,
                    "tee_position":   {"lon": deduped[0][0], "lat": deduped[0][1]},
                    "green_position": {"lon": deduped[-1][0], "lat": deduped[-1][1]},
                    "routing_source": "routing_diagram",
                    "type":           "hole",
                },
            )
            holes.append(hole_feat)

        return holes

    # ── Non-maximum suppression ────────────────────────────────────────────────

    def _nms_by_centroid(self, features: list, min_dist_px: float) -> list:
        """Remove duplicate features whose pixel centroids are closer than min_dist_px."""
        kept = []
        centroids = []
        for f in sorted(features,
                        key=lambda f: f["properties"].get("area_m2", 0),
                        reverse=True):
            cx, cy = self._feature_centroid_px(f)
            too_close = any(_pt_dist((cx, cy), c) < min_dist_px for c in centroids)
            if not too_close:
                kept.append(f)
                centroids.append((cx, cy))
        return kept


# ── Module-level entry points ─────────────────────────────────────────────────

def extract_routing_diagram(
    diagram_path: Path,
    boundary_data: dict,
    output_dir: Path,
    colour_ranges: Optional[dict] = None,
) -> dict:
    """
    Extract golf features from a routing diagram and write GeoJSON files.

    Parameters
    ----------
    diagram_path  : path to diagram image (PNG/JPEG)
    boundary_data : boundary.resolve_boundary() output (provides bbox)
    output_dir    : pipeline output directory
    colour_ranges : optional custom HSV thresholds dict

    Returns
    -------
    dict with keys: fairways, greens, bunkers, water, tees, holes
    Each value is a list of GeoJSON Feature dicts.
    Also returns "paths" key (pointing to the written files).
    """
    output_dir = Path(output_dir)
    extractor  = RoutingDiagramExtractor(
        diagram_path   = diagram_path,
        boundary_data  = boundary_data,
        output_dir     = output_dir,
        colour_ranges  = colour_ranges,
    )

    try:
        result = extractor.extract()
    except Exception as e:
        log.error(f"Routing diagram extraction failed: {e}")
        log.exception(e)
        return {"fairways": [], "greens": [], "bunkers": [],
                "water": [], "tees": [], "holes": []}

    # Write GeoJSON files
    layer_files = {
        "fairways": "routing_fairways.geojson",
        "greens":   "routing_greens.geojson",
        "bunkers":  "routing_bunkers.geojson",
        "water":    "routing_water.geojson",
        "tees":     "routing_tees.geojson",
        "holes":    "routing_holes.geojson",
    }
    result["paths"] = {}
    for layer, filename in layer_files.items():
        path = output_dir / filename
        _write_geojson(result.get(layer, []), path)
        result["paths"][layer] = str(path)

    total = sum(len(result.get(k, [])) for k in layer_files)
    log.info(f"  Routing diagram: {total} features extracted total.")

    # Summary JSON
    summary = {
        "diagram":  str(diagram_path),
        "counts":   {k: len(result.get(k, [])) for k in layer_files},
        "files":    result["paths"],
    }
    (output_dir / "routing_diagram_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return result


def fuse_diagram_features(
    diagram_result: dict,
    output_dir: Path,
    overwrite_layers: Optional[List[str]] = None,
) -> dict:
    """
    Overwrite pipeline GeoJSON files with routing-diagram features.

    Priority: routing_diagram > satellite > OSM

    Parameters
    ----------
    diagram_result    : output of extract_routing_diagram()
    output_dir        : pipeline output directory
    overwrite_layers  : list of layers to overwrite (default: all non-empty)

    Returns
    -------
    dict with per-layer counts after fusion.
    """
    output_dir = Path(output_dir)

    pipeline_files = {
        "fairways": "fairways.geojson",
        "greens":   "greens.geojson",
        "bunkers":  "bunkers.geojson",
        "water":    "water.geojson",
        "tees":     "tees.geojson",
        "holes":    "holes.geojson",
    }

    summary: dict = {}

    for layer, filename in pipeline_files.items():
        diagram_features = diagram_result.get(layer, [])

        if overwrite_layers is not None and layer not in overwrite_layers:
            # Caller explicitly restricted which layers to overwrite
            summary[layer] = {"action": "skipped", "count": 0}
            continue

        if not diagram_features:
            # No diagram features for this layer — preserve existing file
            existing = output_dir / filename
            n = 0
            if existing.exists():
                try:
                    fc = json.loads(existing.read_text(encoding="utf-8"))
                    n  = len(fc.get("features", []))
                except Exception:
                    pass
            summary[layer] = {"action": "preserved", "count": n}
            log.info(f"  Fusion [{layer}]: no diagram features — existing file preserved ({n})")
            continue

        # Write diagram features to the pipeline file
        dest = output_dir / filename
        _write_geojson(diagram_features, dest)
        summary[layer] = {"action": "overwritten", "count": len(diagram_features)}
        log.info(
            f"  Fusion [{layer}]: routing_diagram → {filename} "
            f"({len(diagram_features)} features)"
        )

    # Write fusion manifest
    (output_dir / "routing_diagram_fusion.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary

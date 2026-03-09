"""
routing_diagram_ai.py — Production-grade golf routing diagram extractor.

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
• routing_holes.geojson        ← tee→fairway chain→green LineStrings
• routing_diagram_debug.png    ← colour-coded overlay for QA
• routing_diagram_summary.json

Detection pipeline (8 steps)
-----------------------------
1. HSV colour segmentation — dedicated masks per feature type
2. Morphological cleanup — erode → dilate → remove noise blobs
3. Contour detection — cv2.findContours + area filter
4. Tee detection — approxPolyDP rectangularity test
5. Hole number OCR — pytesseract multi-scale; associate to nearest green
6. Hole routing — tee → fairway centroid chain → green LineStrings
7. Debug visualisation — colour-coded PNG overlay
8. Robustness — handles scorecards, brochures, web images

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
• opencv-python   (cv2)         — in requirements.txt
• numpy, shapely                — in requirements.txt
• pytesseract [OPTIONAL]        — pip install pytesseract + tesseract binary
  If unavailable: hole numbers are skipped gracefully.
"""

import json
import logging
import math
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ── Runtime-optional heavy imports ────────────────────────────────────────────

def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for routing diagram extraction.\n"
            "Install:  pip install opencv-python"
        )


def _try_pytesseract():
    """Return pytesseract module or None if not installed / binary missing."""
    try:
        import pytesseract
        # Honour config.TESSERACT_CMD if set (useful on Windows)
        try:
            import sys, os, importlib.util as _ilu
            _cfg_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "config.py",
            )
            if "config" not in sys.modules or not hasattr(sys.modules["config"], "OVERPASS_URL"):
                _spec = _ilu.spec_from_file_location("config", _cfg_path)
                _mod  = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                sys.modules["config"] = _mod
            import config as _cfg
            _cmd = getattr(_cfg, "TESSERACT_CMD", "")
            if _cmd:
                pytesseract.pytesseract.tesseract_cmd = _cmd
        except Exception:
            pass
        pytesseract.get_tesseract_version()   # confirm binary present
        return pytesseract
    except Exception:
        return None


# ── HSV colour thresholds ─────────────────────────────────────────────────────
# OpenCV HSV convention: H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255]
# Each entry is {"lo": (H, S, V), "hi": (H, S, V)}.
# Multiple ranges per key are OR-merged.
#
# Calibrated for typical golf brochures, scorecard inserts, web PDFs.

_HSV_RANGES: Dict[str, List[Dict]] = {

    # ── Fairway (medium/dark green corridor) ──────────────────────────────────
    "fairway": [
        {"lo": (35, 50,  50),  "hi": (90, 255, 255)},   # primary grass green
        {"lo": (28, 30,  35),  "hi": (92, 200, 200)},   # muted print green
        {"lo": (36, 40,  80),  "hi": (88, 230, 230)},   # colour-washed scan
    ],

    # ── Putting green (lighter, less saturated than fairway) ──────────────────
    "green": [
        {"lo": (35, 30, 120),  "hi": (90, 120, 255)},   # pale / lime green
        {"lo": (85, 30, 100),  "hi": (130, 120, 255)},  # blue-green variant
        {"lo": (36, 20, 150),  "hi": (90,  90, 255)},   # near-white grass
    ],

    # ── Bunkers (sand / cream / pale yellow) ──────────────────────────────────
    "bunker": [
        {"lo": (15, 100, 150), "hi": (40, 255, 255)},   # warm sand
        {"lo": (10,  20, 180), "hi": (40, 120, 255)},   # cream / off-white
        {"lo": (18,  60, 140), "hi": (45, 220, 255)},   # golden sand
    ],

    # ── Water (blue / cyan / navy) ────────────────────────────────────────────
    "water": [
        {"lo": (90,  80,  80), "hi": (140, 255, 255)},  # standard blue
        {"lo": (85,  60,  20), "hi": (135, 255, 160)},  # dark navy
        {"lo": (95,  20, 150), "hi": (140, 100, 255)},  # pale / sky blue
    ],

    # ── Tee markers (coloured rectangles) ────────────────────────────────────
    "tee_red":    [
        {"lo": (0,   130, 100), "hi": (12,  255, 255)},
        {"lo": (168, 130, 100), "hi": (180, 255, 255)},
    ],
    "tee_yellow": [{"lo": (18, 100, 140), "hi": (38, 255, 255)}],
    "tee_blue":   [{"lo": (95,  90,  80), "hi": (130, 255, 255)}],
    "tee_white":  [{"lo": (0,    0, 210), "hi": (180,  25, 255)}],
    "tee_black":  [{"lo": (0,    0,   0), "hi": (180,  60,  50)}],
}

# ── Per-feature area thresholds ───────────────────────────────────────────────
# _AREA_MIN_PX: absolute minimum contour area in pixels (image-size independent
#   down to roughly 800 px wide diagrams).
# _AREA_MAX_FRAC: maximum as fraction of total image area (rejects background).

_AREA_MIN_PX: Dict[str, int] = {
    "fairway": 800,
    "green":   150,
    "bunker":   40,
    "water":   300,
    "tee":      30,
}
_AREA_MAX_FRAC: Dict[str, float] = {
    "fairway": 0.35,
    "green":   0.06,
    "bunker":  0.08,
    "water":   0.35,
    "tee":     0.008,
}

# ── Classification heuristics ─────────────────────────────────────────────────
_GREEN_CIRCULARITY_MIN  = 0.42   # above this → putting green (compact shape)
_GREEN_MAX_AREA_FRAC    = 0.05   # greens can't cover > 5 % of diagram

_TEE_ASPECT_MIN = 1.1            # tees are elongated rectangles
_TEE_ASPECT_MAX = 9.0

# Contour polygon simplification (fraction of arc length)
_SIMPLIFY_EPS_FRAC = 0.005

# ── Debug overlay colours (BGR for OpenCV) ────────────────────────────────────
_DBG_COLOUR = {
    "fairway": (34,  139,  34),   # forest green
    "green":   (0,   200,   0),   # bright green
    "bunker":  (0,   165, 255),   # orange
    "water":   (200,  80,   0),   # deep blue
    "tee":     (0,     0, 220),   # red
    "hole":    (255,   0, 200),   # magenta
    "number":  (0,     0,   0),   # black text
}


# ── Pure-geometry helpers ──────────────────────────────────────────────────────

def _circularity(area: float, perimeter: float) -> float:
    if perimeter < 1e-6:
        return 0.0
    return 4.0 * math.pi * area / (perimeter ** 2)


def _centroid_px(contour) -> Tuple[float, float]:
    cv2 = _require_cv2()
    M = cv2.moments(contour)
    if M["m00"] < 1e-6:
        pts = contour.reshape(-1, 2)
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


def _pt_dist(a: Tuple, b: Tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segment_point_dist(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Distance from point P to line segment A→B."""
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
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


# ── Coordinate transform ───────────────────────────────────────────────────────

def _px_to_lonlat(
    px: float, py: float,
    img_w: int, img_h: int,
    bbox: List[float],
) -> Tuple[float, float]:
    """Map pixel (px, py) to WGS84 (lon, lat). bbox = [min_lon, min_lat, max_lon, max_lat]."""
    min_lon, min_lat, max_lon, max_lat = bbox
    lon = min_lon + (px / img_w) * (max_lon - min_lon)
    lat = max_lat - (py / img_h) * (max_lat - min_lat)
    return lon, lat


# ── GeoJSON helpers ────────────────────────────────────────────────────────────

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
    log.info(f"    {len(features):3d} features  →  {path.name}")


# ── Core extractor class ───────────────────────────────────────────────────────

class RoutingDiagramExtractor:
    """
    Extract golf course features from a routing diagram image.

    Parameters
    ----------
    diagram_path  : Path to input image (PNG / JPEG).
    boundary_data : Output of boundary.resolve_boundary(); provides bbox_wgs84.
    output_dir    : Pipeline output directory; GeoJSON + debug PNG written here.
    colour_ranges : Optional override for HSV colour thresholds.
    """

    def __init__(
        self,
        diagram_path: Path,
        boundary_data: dict,
        output_dir: Path,
        colour_ranges: Optional[Dict] = None,
    ) -> None:
        self.diagram_path   = Path(diagram_path)
        self.boundary_data  = boundary_data
        self.output_dir     = Path(output_dir)
        self.bbox           = boundary_data["bbox_wgs84"]
        self.ranges         = colour_ranges or _HSV_RANGES

        self._img_bgr: Optional[np.ndarray] = None   # original BGR for display
        self._img_rgb: Optional[np.ndarray] = None
        self._img_hsv: Optional[np.ndarray] = None
        self._img_w: int = 0
        self._img_h: int = 0

    # ── Public entry point ─────────────────────────────────────────────────────

    def extract(self) -> dict:
        """
        Run the full 8-step extraction pipeline.

        Returns dict with keys: fairways, greens, bunkers, water, tees, holes.
        """
        cv2 = _require_cv2()

        # Step 1 + image load
        log.info(f"  Loading:   {self.diagram_path.name}")
        self._load_image()

        # Step 1 — colour masks + Step 2 — morphological cleanup
        log.info("  Step 1-2:  HSV segmentation + morphological cleanup")
        fairways = self._detect_layer("fairway")
        greens   = self._detect_layer("green")
        bunkers  = self._detect_layer("bunker")
        water    = self._detect_layer("water")

        # Resolve fairway/green ambiguity: re-classify overlapping regions
        fairways, greens = self._resolve_fairway_green_overlap(fairways, greens)

        # Step 4 — tee detection
        log.info("  Step 4:    Tee detection (approxPolyDP rectangularity)")
        tees = self._detect_tees()

        log.info(
            f"  Counts  — fairways:{len(fairways)}  greens:{len(greens)}  "
            f"bunkers:{len(bunkers)}  water:{len(water)}  tees:{len(tees)}"
        )

        # Step 5 — OCR
        log.info("  Step 5:    OCR — hole number detection")
        hole_positions = self._detect_hole_numbers()
        log.info(f"  OCR found  {len(hole_positions)} hole number(s)")

        # Hole association + routing
        log.info("  Step 5-6:  Associating holes + building routing")
        hole_map = self._associate_holes(hole_positions, tees, greens, fairways)
        holes    = self._construct_routing(hole_map, tees, greens, fairways)
        log.info(f"  Routing:   {len(holes)} hole route(s)")

        # Stamp hole_number onto matched features
        self._stamp_hole_numbers(hole_map, tees, greens, fairways)

        result = {
            "fairways": fairways,
            "greens":   greens,
            "bunkers":  bunkers,
            "water":    water,
            "tees":     tees,
            "holes":    holes,
        }

        # Step 7 — debug overlay
        log.info("  Step 7:    Writing debug overlay")
        self._write_debug_overlay(result)

        return result

    # ── Step 1: image loading ──────────────────────────────────────────────────

    def _load_image(self) -> None:
        cv2 = _require_cv2()
        img_bgr = cv2.imread(str(self.diagram_path))
        if img_bgr is None:
            raise ValueError(f"Cannot open image: {self.diagram_path}")

        # Upscale very small diagrams so morphology and OCR work better
        h, w = img_bgr.shape[:2]
        if w < 800:
            scale    = 800 / w
            img_bgr  = cv2.resize(img_bgr, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_CUBIC)
            log.info(f"  Upscaled {w}×{h} → {img_bgr.shape[1]}×{img_bgr.shape[0]} px")

        self._img_bgr  = img_bgr
        self._img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self._img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        self._img_h, self._img_w = img_bgr.shape[:2]
        log.info(f"  Image:     {self._img_w} × {self._img_h} px")

    # ── Step 1-2: mask helpers ─────────────────────────────────────────────────

    def _build_mask(self, range_key: str) -> np.ndarray:
        """
        OR-merge all HSV ranges for range_key into a single uint8 binary mask.
        """
        cv2  = _require_cv2()
        hsv  = self._img_hsv
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for r in self.ranges.get(range_key, []):
            lo = np.array(r["lo"], dtype=np.uint8)
            hi = np.array(r["hi"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        return mask

    def _clean_mask(
        self,
        mask: np.ndarray,
        erode_k: int = 2,
        dilate_k: int = 4,
        close_k:  int = 8,
        min_blob: int = 20,
    ) -> np.ndarray:
        """
        Step 2 — Morphological cleanup.
        1. Erode  (remove thin noise)
        2. Dilate (restore / bridge small gaps)
        3. Close  (fill interior holes in blobs)
        4. Remove blobs below min_blob area.
        """
        cv2 = _require_cv2()

        def kernel(k):
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        out = cv2.erode(mask,  kernel(erode_k),  iterations=1)
        out = cv2.dilate(out,  kernel(dilate_k), iterations=1)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel(close_k))

        # Remove blobs smaller than min_blob px²
        if min_blob > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
            for lbl in range(1, n_labels):
                if stats[lbl, cv2.CC_STAT_AREA] < min_blob:
                    out[labels == lbl] = 0

        return out

    def _area_bounds(self, feature_type: str) -> Tuple[float, float]:
        total = self._img_w * self._img_h
        min_a = _AREA_MIN_PX.get(feature_type, 50)
        max_a = _AREA_MAX_FRAC.get(feature_type, 0.30) * total
        return float(min_a), float(max_a)

    def _find_contours(
        self,
        mask: np.ndarray,
        min_area: float,
        max_area: float,
    ) -> list:
        """Step 3 — extract and filter contours by area."""
        cv2 = _require_cv2()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return [c for c in cnts if min_area <= cv2.contourArea(c) <= max_area]

    # ── Step 1-3: per-layer detection ──────────────────────────────────────────

    def _detect_layer(self, feature_type: str) -> List[dict]:
        """
        Steps 1-3 for a single layer:
          build HSV mask → morphological cleanup → contour detection → GeoJSON.
        """
        min_a, max_a = self._area_bounds(feature_type)

        # Choose cleanup aggressiveness by feature size
        if feature_type == "fairway":
            mask = self._clean_mask(self._build_mask("fairway"),
                                    erode_k=3, dilate_k=6, close_k=12, min_blob=200)
        elif feature_type == "green":
            mask = self._clean_mask(self._build_mask("green"),
                                    erode_k=2, dilate_k=3, close_k=7,  min_blob=50)
        elif feature_type == "bunker":
            mask = self._clean_mask(self._build_mask("bunker"),
                                    erode_k=2, dilate_k=3, close_k=5,  min_blob=20)
        elif feature_type == "water":
            mask = self._clean_mask(self._build_mask("water"),
                                    erode_k=3, dilate_k=5, close_k=10, min_blob=100)
        else:
            mask = self._clean_mask(self._build_mask(feature_type))

        contours = self._find_contours(mask, min_a, max_a)
        features = []
        for c in contours:
            feat = self._contour_to_feature(c, feature_type)
            if feat:
                features.append(feat)
        return features

    # ── Green / fairway overlap resolution ────────────────────────────────────

    def _resolve_fairway_green_overlap(
        self,
        fairways: List[dict],
        greens: List[dict],
    ) -> Tuple[List[dict], List[dict]]:
        """
        Re-classify ambiguous green-coloured regions using circularity + area.

        Both fairways and greens share green hues; putting greens are compact
        (high circularity) while fairways are elongated (low circularity).
        Merge all, then re-split.
        """
        cv2 = _require_cv2()
        total_area  = self._img_w * self._img_h
        green_max_a = _GREEN_MAX_AREA_FRAC * total_area

        all_green_mask = self._build_mask("fairway")
        green_mask     = self._build_mask("green")
        all_green_mask = _require_cv2().bitwise_or(all_green_mask, green_mask)
        all_green_mask = self._clean_mask(all_green_mask,
                                          erode_k=2, dilate_k=5, close_k=11, min_blob=50)

        min_a_fw, max_a_fw = self._area_bounds("fairway")
        min_a_g  = _AREA_MIN_PX["green"]
        contours = self._find_contours(all_green_mask, float(min_a_g), float(max_a_fw))

        new_fairways: List[dict] = []
        new_greens:   List[dict] = []

        for c in contours:
            area = float(cv2.contourArea(c))
            peri = float(cv2.arcLength(c, closed=True))
            circ = _circularity(area, peri)
            is_green = (circ >= _GREEN_CIRCULARITY_MIN) and (area <= green_max_a)

            feat = self._contour_to_feature(c, "green" if is_green else "fairway")
            if feat is None:
                continue
            if is_green:
                new_greens.append(feat)
            else:
                new_fairways.append(feat)

        return new_fairways, new_greens

    # ── Contour → GeoJSON polygon ──────────────────────────────────────────────

    def _contour_to_feature(
        self,
        contour,
        feature_type: str,
        extra_props: Optional[dict] = None,
    ) -> Optional[dict]:
        """
        Convert pixel-space contour to a WGS84 GeoJSON Polygon feature.
        Returns None if fewer than 3 vertices survive simplification.
        """
        cv2 = _require_cv2()
        arc = float(cv2.arcLength(contour, closed=True))
        eps = max(1.5, _SIMPLIFY_EPS_FRAC * arc)
        simplified = cv2.approxPolyDP(contour, eps, closed=True).reshape(-1, 2)
        if len(simplified) < 3:
            return None

        coords = [
            list(_px_to_lonlat(float(x), float(y), self._img_w, self._img_h, self.bbox))
            for x, y in simplified
        ]
        coords.append(coords[0])   # close ring

        # Optionally smooth with Shapely
        geom: dict
        try:
            from shapely.geometry import Polygon, mapping
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_valid or poly.is_empty:
                return None
            poly = poly.simplify(2.0 / 111_000, preserve_topology=True)
            if poly.is_empty:
                return None
            geom = mapping(poly)
        except Exception:
            geom = {"type": "Polygon", "coordinates": [coords]}

        # Centroid + approximate real-world area
        cx, cy = _centroid_px(contour)
        px_area = float(cv2.contourArea(contour))
        img_area = self._img_w * self._img_h
        bbox_m2 = (
            _haversine_m(self.bbox[0], (self.bbox[1] + self.bbox[3]) / 2,
                         self.bbox[2], (self.bbox[1] + self.bbox[3]) / 2)
            * _haversine_m((self.bbox[0] + self.bbox[2]) / 2, self.bbox[1],
                           (self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])
        )
        area_m2 = round(px_area / img_area * bbox_m2, 1) if img_area else 0.0

        fid   = str(uuid.uuid4())
        props = dict(extra_props or {})
        props.update({
            "id":          fid,
            "type":        feature_type,
            "source":      "routing_diagram",
            "area_m2":     area_m2,
            "centroid_px": [round(cx, 1), round(cy, 1)],
        })
        return {"type": "Feature", "geometry": geom, "properties": props}

    # ── Step 4: tee detection ──────────────────────────────────────────────────

    def _detect_tees(self) -> List[dict]:
        """
        Detect tee boxes: small, roughly rectangular coloured shapes.

        Algorithm:
        1. OR-merge all tee colour masks.
        2. Morphological cleanup.
        3. Find contours.
        4. For each contour, run cv2.approxPolyDP and check:
           - vertex count ∈ [4, 8]  (rectangle-like)
           - aspect ratio  ∈ [_TEE_ASPECT_MIN, _TEE_ASPECT_MAX]
        5. NMS by centroid proximity.
        """
        cv2 = _require_cv2()
        tee_keys = ["tee_red", "tee_yellow", "tee_blue", "tee_white", "tee_black"]
        merged   = np.zeros((self._img_h, self._img_w), dtype=np.uint8)
        for k in tee_keys:
            merged = cv2.bitwise_or(merged, self._build_mask(k))

        mask     = self._clean_mask(merged, erode_k=1, dilate_k=2, close_k=4, min_blob=15)
        min_a, max_a = self._area_bounds("tee")
        contours = self._find_contours(mask, min_a, max_a)

        features: List[dict] = []
        for c in contours:
            # Rectangularity check via approxPolyDP
            arc  = float(cv2.arcLength(c, closed=True))
            eps  = 0.04 * arc                     # tighter epsilon for rectangles
            poly = cv2.approxPolyDP(c, eps, closed=True)
            n    = len(poly)
            if n < 4 or n > 8:
                continue                           # not rectangle-like

            rect     = cv2.minAreaRect(c)
            w_r, h_r = rect[1]
            if w_r < 1 or h_r < 1:
                continue
            aspect = max(w_r, h_r) / min(w_r, h_r)
            if not (_TEE_ASPECT_MIN <= aspect <= _TEE_ASPECT_MAX):
                continue

            feat = self._contour_to_feature(c, "tee")
            if feat:
                features.append(feat)

        return self._nms_by_centroid(features, min_dist_px=_TEE_NMS_PX)

    # ── Step 5: OCR ───────────────────────────────────────────────────────────

    def _detect_hole_numbers(self) -> List[dict]:
        """
        Use pytesseract to locate digit labels 1–18 in the diagram.

        Multi-scale + multi-threshold preprocessing maximises recall on:
        - printed / scanned diagrams (low contrast)
        - web images with anti-aliased text
        - coloured backgrounds (adaptive thresholding)

        Returns list of {"number": int, "x": float, "y": float, "conf": float}.
        """
        tess = _try_pytesseract()
        if tess is None:
            log.warning("  pytesseract unavailable — hole OCR skipped.")
            log.warning("  Install:  pip install pytesseract  +  tesseract binary")
            return []

        cv2  = _require_cv2()
        gray = cv2.cvtColor(self._img_bgr, cv2.COLOR_BGR2GRAY)
        candidates: Dict[int, dict] = {}   # number → best hit

        def _record(num, cx, cy, conf, scale):
            if num < 1 or num > 18:
                return
            if conf < 25:
                return
            px = cx / scale
            py = cy / scale
            if num not in candidates or conf > candidates[num]["conf"]:
                candidates[num] = {"number": num, "x": px, "y": py, "conf": conf}

        for scale in [1.0, 1.5, 2.0]:
            w   = int(self._img_w * scale)
            h   = int(self._img_h * scale)
            img = cv2.resize(gray, (w, h), interpolation=cv2.INTER_CUBIC) \
                  if scale != 1.0 else gray.copy()

            # Preprocess variants to handle different printing styles
            _, otsu    = cv2.threshold(img, 0, 255,
                                       cv2.THRESH_BINARY     + cv2.THRESH_OTSU)
            _, otsu_inv = cv2.threshold(img, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            adapt      = cv2.adaptiveThreshold(img, 255,
                                               cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                               cv2.THRESH_BINARY, 15, 4)

            ocr_cfg    = "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789"

            for src in [img, otsu, otsu_inv, adapt]:
                try:
                    data = tess.image_to_data(
                        src,
                        config=ocr_cfg,
                        output_type=tess.Output.DICT,
                    )
                except Exception as exc:
                    log.debug(f"tesseract error: {exc}")
                    continue

                for i, text in enumerate(data["text"]):
                    text = (text or "").strip()
                    if not text.isdigit():
                        continue
                    num = int(text)
                    try:
                        conf = float(data["conf"][i])
                    except (TypeError, ValueError):
                        continue
                    cx = data["left"][i]  + data["width"][i]  / 2
                    cy = data["top"][i]   + data["height"][i] / 2
                    _record(num, cx, cy, conf, scale)

        return sorted(candidates.values(), key=lambda d: d["number"])

    # ── Hole association ───────────────────────────────────────────────────────

    def _get_centroid_px(self, feature: dict) -> Tuple[float, float]:
        cp = feature.get("properties", {}).get("centroid_px")
        if cp:
            return float(cp[0]), float(cp[1])
        min_lon, min_lat, max_lon, max_lat = self.bbox
        geom   = feature.get("geometry", {})
        coords = geom.get("coordinates", [[]])[0]
        if not coords:
            return 0.0, 0.0
        xs = [((p[0] - min_lon) / (max_lon - min_lon)) * self._img_w for p in coords]
        ys = [((max_lat - p[1]) / (max_lat - min_lat)) * self._img_h for p in coords]
        return sum(xs) / len(xs), sum(ys) / len(ys)

    def _associate_holes(
        self,
        hole_positions: List[dict],
        tees: List[dict],
        greens: List[dict],
        fairways: List[dict],
    ) -> dict:
        """
        Match OCR hole numbers to nearest unassigned tee / green / fairway.

        Distance limits (as fraction of max image dimension):
          tee      — 20 %
          green    — 45 %
          fairway  — 50 % (measured to T→G segment, not point)

        Returns {hole_num: {tee_id, green_id, fairway_id, tee_px, green_px, number_px}}.
        """
        diag = math.hypot(self._img_w, self._img_h)

        def _precompute(features):
            return [(self._get_centroid_px(f), f["properties"]["id"]) for f in features]

        tee_pts  = _precompute(tees)
        grn_pts  = _precompute(greens)
        fw_pts   = _precompute(fairways)

        used_t   = set()
        used_g   = set()
        used_fw  = set()
        hole_map: dict = {}

        def _nearest(pts, used, ox, oy, max_d):
            best_d, best_id, best_pt = float("inf"), None, (ox, oy)
            for pt, fid in pts:
                if fid in used:
                    continue
                d = _pt_dist((ox, oy), pt)
                if d < best_d and d < max_d:
                    best_d, best_id, best_pt = d, fid, pt
            return best_id, best_pt

        for hp in hole_positions:
            hx, hy = hp["x"], hp["y"]
            num    = hp["number"]

            t_id, t_px = _nearest(tee_pts, used_t,  hx, hy, 0.20 * diag)
            g_id, g_px = _nearest(grn_pts, used_g,  hx, hy, 0.45 * diag)

            # Fairway nearest to the T→G axis
            f_id = None
            if t_id or g_id:
                best_fd, best_fid = float("inf"), None
                tx, ty = t_px
                gx, gy = g_px
                for (fx, fy), fid in fw_pts:
                    if fid in used_fw:
                        continue
                    d = _segment_point_dist(fx, fy, tx, ty, gx, gy)
                    if d < best_fd and d < 0.50 * diag:
                        best_fd, best_fid = d, fid
                f_id = best_fid

            hole_map[num] = {
                "tee_id":    t_id,
                "green_id":  g_id,
                "fairway_id": f_id,
                "tee_px":    t_px,
                "green_px":  g_px,
                "number_px": (hx, hy),
            }
            if t_id:  used_t.add(t_id)
            if g_id:  used_g.add(g_id)
            if f_id:  used_fw.add(f_id)

        # Infer additional holes when OCR found fewer than half
        if len(hole_map) < max(1, len(greens) // 2):
            hole_map = self._infer_remaining_holes(
                hole_map, tees, greens, fairways,
                used_t, used_g, used_fw,
            )

        return hole_map

    def _infer_remaining_holes(
        self,
        hole_map: dict,
        tees: List[dict],
        greens: List[dict],
        fairways: List[dict],
        used_t: set,
        used_g: set,
        used_fw: set,
    ) -> dict:
        """Pair remaining tees → greens by nearest-neighbour when OCR < ½ holes."""
        unassigned_t  = [f for f in tees    if f["properties"]["id"] not in used_t]
        unassigned_g  = [f for f in greens  if f["properties"]["id"] not in used_g]
        unassigned_fw = [f for f in fairways if f["properties"]["id"] not in used_fw]
        next_num      = max(hole_map.keys(), default=0) + 1

        for tee in unassigned_t:
            if next_num > 18 or not unassigned_g:
                break
            tpx = self._get_centroid_px(tee)
            # nearest green
            best_g = min(unassigned_g, key=lambda g: _pt_dist(tpx, self._get_centroid_px(g)))
            gpx    = self._get_centroid_px(best_g)

            # nearest fairway along T→G
            best_fw = None
            if unassigned_fw:
                best_fw = min(
                    unassigned_fw,
                    key=lambda f: _segment_point_dist(
                        *self._get_centroid_px(f), *tpx, *gpx
                    ),
                )

            hole_map[next_num] = {
                "tee_id":     tee["properties"]["id"],
                "green_id":   best_g["properties"]["id"],
                "fairway_id": best_fw["properties"]["id"] if best_fw else None,
                "tee_px":     tpx,
                "green_px":   gpx,
                "number_px":  tpx,
            }
            used_t.add(tee["properties"]["id"])
            used_g.add(best_g["properties"]["id"])
            if best_fw:
                used_fw.add(best_fw["properties"]["id"])

            unassigned_g  = [g for g in unassigned_g  if g["properties"]["id"] not in used_g]
            unassigned_fw = [f for f in unassigned_fw if f["properties"]["id"] not in used_fw]
            next_num     += 1

        return hole_map

    def _stamp_hole_numbers(self, hole_map, tees, greens, fairways):
        for hole_num, asgn in hole_map.items():
            for layer_features, id_key in [
                (tees,     "tee_id"),
                (greens,   "green_id"),
                (fairways, "fairway_id"),
            ]:
                fid = asgn.get(id_key)
                if not fid:
                    continue
                for f in layer_features:
                    if f["properties"].get("id") == fid:
                        f["properties"]["hole_number"] = hole_num

    # ── Step 6: routing construction ──────────────────────────────────────────

    def _construct_routing(
        self,
        hole_map: dict,
        tees: List[dict],
        greens: List[dict],
        fairways: List[dict],
    ) -> List[dict]:
        """
        Build tee → fairway centroid chain → green LineString for each hole.

        The fairway centroid is a single waypoint — enough to establish the
        corridor while keeping the GeoJSON compact. Holes with no tee AND no
        green are omitted.
        """
        def _by_id(features, fid):
            return next((f for f in features if f["properties"]["id"] == fid), None)

        def _wgs84_centroid(feature) -> Optional[Tuple[float, float]]:
            if feature is None:
                return None
            geom   = feature.get("geometry", {})
            coords = geom.get("coordinates", [[]])[0]
            if not coords:
                return None
            lons = [p[0] for p in coords if len(p) >= 2]
            lats = [p[1] for p in coords if len(p) >= 2]
            if not lons:
                return None
            return sum(lons) / len(lons), sum(lats) / len(lats)

        holes: List[dict] = []

        for num in sorted(hole_map.keys()):
            asgn  = hole_map[num]
            tee   = _by_id(tees,     asgn.get("tee_id"))
            green = _by_id(greens,   asgn.get("green_id"))
            fw    = _by_id(fairways, asgn.get("fairway_id"))

            if tee is None and green is None:
                continue

            route: List[List[float]] = []
            for feat in [tee, fw, green]:
                c = _wgs84_centroid(feat)
                if c:
                    route.append(list(c))

            # De-duplicate consecutive identical points
            deduped: List[List[float]] = []
            for pt in route:
                if not deduped or pt != deduped[-1]:
                    deduped.append(pt)
            if len(deduped) < 2:
                continue

            dist_m  = sum(
                _haversine_m(deduped[i][0], deduped[i][1],
                             deduped[i+1][0], deduped[i+1][1])
                for i in range(len(deduped) - 1)
            )
            dist_yd = round(dist_m * 1.09361)
            par     = 3 if dist_yd < 250 else (5 if dist_yd > 470 else 4)

            holes.append(_make_feature(
                {"type": "LineString", "coordinates": deduped},
                {
                    "hole_number":    num,
                    "par":            par,
                    "distance_m":     round(dist_m, 1),
                    "distance_yards": dist_yd,
                    "length_m":       round(dist_m, 1),
                    "length_yards":   dist_yd,
                    "tee_position":   {"lon": deduped[0][0],  "lat": deduped[0][1]},
                    "green_position": {"lon": deduped[-1][0], "lat": deduped[-1][1]},
                    "routing_source": "routing_diagram",
                    "type":           "hole",
                },
            ))

        return holes

    # ── Step 7: debug visualisation ───────────────────────────────────────────

    def _write_debug_overlay(self, result: dict) -> None:
        """
        Draw detected features on a copy of the original image and save as
        routing_diagram_debug.png.

        Draws:
        • Filled + outlined polygons per feature type
        • Hole number labels at green centroids
        • Hole routing lines
        """
        try:
            cv2 = _require_cv2()
            overlay = self._img_bgr.copy()
            alpha   = 0.35                   # transparency of filled polygons

            def _lonlat_ring_to_px(ring):
                """Convert WGS84 ring back to integer pixel coords."""
                min_lon, min_lat, max_lon, max_lat = self.bbox
                pts = []
                for p in ring:
                    if len(p) < 2:
                        continue
                    lon, lat = p[0], p[1]
                    x = int((lon - min_lon) / (max_lon - min_lon) * self._img_w)
                    y = int((max_lat - lat) / (max_lat - min_lat) * self._img_h)
                    pts.append([x, y])
                return np.array(pts, dtype=np.int32) if len(pts) >= 2 else None

            def _draw_polygon(feat, colour, label=None):
                geom = feat.get("geometry", {})
                gtype = geom.get("type", "")
                if gtype == "Polygon":
                    rings = geom.get("coordinates", [])
                elif gtype == "MultiPolygon":
                    rings = [r for poly in geom.get("coordinates", []) for r in poly]
                else:
                    return

                for ring in rings[:1]:   # outer ring only for performance
                    pts = _lonlat_ring_to_px(ring)
                    if pts is None or len(pts) < 3:
                        continue
                    # Semi-transparent fill
                    fill_img = overlay.copy()
                    cv2.fillPoly(fill_img, [pts], colour)
                    cv2.addWeighted(fill_img, alpha, overlay, 1 - alpha, 0, overlay)
                    # Outline
                    cv2.polylines(overlay, [pts], isClosed=True, color=colour, thickness=2)

                if label:
                    cp = feat.get("properties", {}).get("centroid_px")
                    if cp:
                        cx, cy = int(cp[0]), int(cp[1])
                        cv2.putText(overlay, label, (cx - 8, cy + 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    _DBG_COLOUR["number"], 2, cv2.LINE_AA)

            # Draw by layer (back to front)
            for feat in result.get("water",    []): _draw_polygon(feat, _DBG_COLOUR["water"])
            for feat in result.get("bunkers",  []): _draw_polygon(feat, _DBG_COLOUR["bunker"])
            for feat in result.get("fairways", []): _draw_polygon(feat, _DBG_COLOUR["fairway"])
            for feat in result.get("greens",   []):
                hn = feat.get("properties", {}).get("hole_number")
                _draw_polygon(feat, _DBG_COLOUR["green"], label=str(hn) if hn else None)
            for feat in result.get("tees",     []): _draw_polygon(feat, _DBG_COLOUR["tee"])

            # Draw hole routing lines
            min_lon, min_lat, max_lon, max_lat = self.bbox
            for hole in result.get("holes", []):
                coords = hole.get("geometry", {}).get("coordinates", [])
                pts    = _lonlat_ring_to_px(coords)
                if pts is not None and len(pts) >= 2:
                    cv2.polylines(overlay, [pts], isClosed=False,
                                  color=_DBG_COLOUR["hole"], thickness=2)
                    # Arrowhead at green end
                    if len(pts) >= 2:
                        p1, p2 = pts[-2].tolist(), pts[-1].tolist()
                        cv2.arrowedLine(overlay, tuple(p1), tuple(p2),
                                        _DBG_COLOUR["hole"], 2, tipLength=0.3)

            # Legend
            legend_y = 20
            for label, colour in [
                ("fairway", _DBG_COLOUR["fairway"]),
                ("green",   _DBG_COLOUR["green"]),
                ("bunker",  _DBG_COLOUR["bunker"]),
                ("water",   _DBG_COLOUR["water"]),
                ("tee",     _DBG_COLOUR["tee"]),
                ("routing", _DBG_COLOUR["hole"]),
            ]:
                cv2.rectangle(overlay, (8, legend_y - 10), (24, legend_y + 4), colour, -1)
                cv2.putText(overlay, label, (28, legend_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(overlay, label, (28, legend_y + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                legend_y += 22

            out_path = self.output_dir / "routing_diagram_debug.png"
            cv2.imwrite(str(out_path), overlay)
            log.info(f"  Debug PNG: {out_path.name}")

        except Exception as exc:
            log.warning(f"  Debug overlay failed (non-critical): {exc}")

    # ── NMS ───────────────────────────────────────────────────────────────────

    def _nms_by_centroid(self, features: List[dict], min_dist_px: float) -> List[dict]:
        """Remove duplicates whose pixel centroids are within min_dist_px."""
        kept: List[dict] = []
        centroids: List[Tuple[float, float]] = []
        # Process largest first so dominant detection wins
        for f in sorted(features,
                         key=lambda x: x["properties"].get("area_m2", 0),
                         reverse=True):
            cx, cy = self._get_centroid_px(f)
            if not any(_pt_dist((cx, cy), c) < min_dist_px for c in centroids):
                kept.append(f)
                centroids.append((cx, cy))
        return kept


# ── Module-level entry points ──────────────────────────────────────────────────

def extract_routing_diagram(
    diagram_path: Path,
    boundary_data: dict,
    output_dir: Path,
    colour_ranges: Optional[Dict] = None,
) -> dict:
    """
    Extract golf features from a routing diagram and write GeoJSON + debug PNG.

    Parameters
    ----------
    diagram_path  : path to diagram image (PNG / JPEG)
    boundary_data : boundary.resolve_boundary() output (provides bbox_wgs84)
    output_dir    : pipeline output directory
    colour_ranges : optional dict overriding default HSV thresholds

    Returns
    -------
    dict with keys: fairways, greens, bunkers, water, tees, holes, paths
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
    except Exception as exc:
        log.error(f"Routing diagram extraction failed: {exc}")
        log.exception(exc)
        return {"fairways": [], "greens": [], "bunkers": [],
                "water": [], "tees": [], "holes": []}

    # Write per-layer GeoJSON files
    layer_files = {
        "fairways": "routing_fairways.geojson",
        "greens":   "routing_greens.geojson",
        "bunkers":  "routing_bunkers.geojson",
        "water":    "routing_water.geojson",
        "tees":     "routing_tees.geojson",
        "holes":    "routing_holes.geojson",
    }
    log.info("  Writing GeoJSON outputs:")
    result["paths"] = {}
    for layer, filename in layer_files.items():
        path = output_dir / filename
        _write_geojson(result.get(layer, []), path)
        result["paths"][layer] = str(path)

    total = sum(len(result.get(k, [])) for k in layer_files)
    log.info(f"  Total features extracted: {total}")

    # Summary JSON
    summary = {
        "diagram": str(diagram_path),
        "counts":  {k: len(result.get(k, [])) for k in layer_files},
        "files":   result["paths"],
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
    diagram_result   : output of extract_routing_diagram()
    output_dir       : pipeline output directory
    overwrite_layers : layers to overwrite (default: all non-empty)

    Returns
    -------
    dict — per-layer {action, count} summary.
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
        if overwrite_layers is not None and layer not in overwrite_layers:
            summary[layer] = {"action": "skipped", "count": 0}
            continue

        diagram_features = diagram_result.get(layer, [])

        if not diagram_features:
            existing = output_dir / filename
            n = 0
            if existing.exists():
                try:
                    fc = json.loads(existing.read_text(encoding="utf-8"))
                    n  = len(fc.get("features", []))
                except Exception:
                    pass
            summary[layer] = {"action": "preserved", "count": n}
            log.info(f"  Fusion [{layer:9s}]: no diagram features — preserved ({n})")
            continue

        dest = output_dir / filename
        _write_geojson(diagram_features, dest)
        summary[layer] = {"action": "overwritten", "count": len(diagram_features)}
        log.info(
            f"  Fusion [{layer:9s}]: routing_diagram → {filename} "
            f"({len(diagram_features)} features)"
        )

    (output_dir / "routing_diagram_fusion.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    return summary

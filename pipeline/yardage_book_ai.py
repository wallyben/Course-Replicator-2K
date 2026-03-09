"""
yardage_book_ai.py — Yardage book and hole flyover diagram extractor.

Parses scanned yardage books, hole flyover diagrams, and scorecards to extract
accurate hole routing information for the Course Replicator 2K pipeline.

Pipeline position: after satellite vision, before feature fusion.

Inputs:
    assets/yardage_books/*.{jpg,jpeg,png,pdf}  (user-supplied)

Outputs (per run, in output/<course>/):
    yardage_holes.geojson     — per-hole routing GeoJSON (LineStrings)
    yardage_fairways.geojson  — detected fairway polygons
    yardage_greens.geojson    — detected green polygons
    yardage_bunkers.geojson   — detected bunker polygons
    yardage_water.geojson     — detected water hazard polygons
    yardage_debug_h{n}.png    — per-hole debug overlay (optional)

Usage:
    from pipeline.yardage_book_ai import YardageBookExtractor
    extractor = YardageBookExtractor(assets_dir, output_dir, debug=True)
    result = extractor.run()
    # result["holes"] → list of per-hole dicts with features + routing
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import importlib.util as _ilu
_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py"
)
if "config" not in sys.modules or not hasattr(sys.modules["config"], "OVERPASS_URL"):
    _spec = _ilu.spec_from_file_location("config", _CONFIG_PATH)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["config"] = _mod
import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

_ENABLED        = getattr(config, "YARDAGE_BOOK_ENABLED", True)
_MIN_CONFIDENCE = getattr(config, "YARDAGE_BOOK_MIN_CONFIDENCE", 0.35)
_ASSETS_DIR     = Path(getattr(config, "ASSETS_DIR", "assets"))
_YARDAGE_DIR    = Path(getattr(config, "YARDAGE_DIR", str(_ASSETS_DIR / "yardage_books")))

# Supported input image extensions
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# HSV colour ranges for yardage book diagram features
# Yardage books typically use standardised colours (vivid print colours)
_HSV_RANGES: Dict[str, List[Dict]] = {
    "fairway": [
        {"lo": (35, 60,  80),  "hi": (90, 255, 255)},   # vivid green
        {"lo": (28, 40,  60),  "hi": (92, 180, 210)},   # muted print green
    ],
    "green": [
        {"lo": (36, 30, 130),  "hi": (90, 110, 255)},   # lighter green (putting surface)
        {"lo": (85, 25, 100),  "hi": (130, 110, 255)},  # blue-green variant
    ],
    "bunker": [
        {"lo": (15, 100, 150), "hi": (40, 255, 255)},   # golden / sandy yellow
        {"lo": (10,  20, 180), "hi": (42, 110, 255)},   # cream / light yellow
        {"lo": ( 0,   0, 215), "hi": (30,  40, 255)},   # near-white / off-white
    ],
    "water": [
        {"lo": (90,  80,  80), "hi": (140, 255, 255)},  # standard blue
        {"lo": (85,  50,  20), "hi": (135, 255, 160)},  # dark navy
        {"lo": (175, 60,  60), "hi": (180, 255, 255)},  # wrap-around cyan
    ],
    "tee": [
        {"lo": ( 0,  80, 120), "hi": (15, 255, 255)},   # red/orange tee marker
        {"lo": (165, 80, 120), "hi": (180, 255, 255)},  # wrap-around red
        {"lo": (100, 80, 120), "hi": (140, 255, 255)},  # blue tee marker
    ],
}

# Minimum pixel area (at 200 DPI equivalent scale) to accept a contour
_AREA_MIN_PX: Dict[str, int] = {
    "fairway": 600,
    "green":   120,
    "bunker":   30,
    "water":   200,
    "tee":      20,
}

# Circularity threshold separating greens from fairways (same colour hue)
_GREEN_CIRCULARITY_MIN = 0.42

# OCR confidence — hole numbers must be numeric digits
_OCR_CONF_MIN = 40  # Tesseract confidence threshold


# ─── Public API ───────────────────────────────────────────────────────────────

class YardageBookExtractor:
    """
    Extract golf hole features from yardage book images.

    Parameters
    ----------
    assets_dir : Path-like
        Directory containing yardage_books/ sub-folder with diagram images.
    output_dir : Path-like
        Where GeoJSON outputs are written.
    debug : bool
        Write per-hole debug overlay PNGs.
    """

    def __init__(
        self,
        assets_dir: Path,
        output_dir: Path,
        debug: bool = False,
    ) -> None:
        self.assets_dir  = Path(assets_dir)
        self.yardage_dir = self.assets_dir / "yardage_books"
        self.output_dir  = Path(output_dir)
        self.debug       = debug
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(self) -> Dict:
        """
        Process all yardage book images and produce GeoJSON outputs.

        Returns a dict with:
            "holes"    : list of per-hole dicts
            "geojson"  : merged FeatureCollection covering all holes
            "skipped"  : reason string if module disabled or no images found
        """
        if not _ENABLED:
            log.info("YardageBookAI: disabled in config")
            return {"skipped": "disabled", "holes": [], "geojson": _empty_fc()}

        images = self._collect_images()
        if not images:
            log.info("YardageBookAI: no yardage book images found in %s", self.yardage_dir)
            return {"skipped": "no_images", "holes": [], "geojson": _empty_fc()}

        log.info("YardageBookAI: found %d image(s) to process", len(images))

        all_holes: List[Dict] = []
        for img_path in sorted(images):
            try:
                holes = self._process_image(img_path)
                all_holes.extend(holes)
                log.info("  %s → %d hole(s) extracted", img_path.name, len(holes))
            except Exception as exc:
                log.warning("  %s → error: %s", img_path.name, exc)

        # Deduplicate by hole number (keep highest-confidence entry)
        all_holes = _deduplicate_holes(all_holes)

        # Write GeoJSON outputs
        self._write_geojsons(all_holes)

        merged_fc = _holes_to_featurecollection(all_holes)
        return {"holes": all_holes, "geojson": merged_fc}

    # ── Image collection ─────────────────────────────────────────────────────

    def _collect_images(self) -> List[Path]:
        """Return all supported image files from yardage_books/."""
        self.yardage_dir.mkdir(parents=True, exist_ok=True)
        return [
            p for p in sorted(self.yardage_dir.iterdir())
            if p.suffix.lower() in _IMAGE_EXTS
        ]

    # ── Per-image processing ─────────────────────────────────────────────────

    def _process_image(self, img_path: Path) -> List[Dict]:
        """
        Attempt to interpret one image as a yardage book diagram.

        Strategy:
          1. Load and normalise image size.
          2. Detect layout (single-hole or multi-hole grid).
          3. For each detected panel, extract features and OCR hole number.
          4. Return per-hole dicts.
        """
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise ValueError(f"Cannot read image: {img_path}")

        bgr = _normalise_image(bgr)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Try to split multi-hole grids; fall back to treating whole image as one hole
        panels = self._split_panels(bgr)
        if not panels:
            panels = [("unknown", bgr)]

        holes: List[Dict] = []
        for panel_id, panel_bgr in panels:
            try:
                hole_dict = self._extract_hole(panel_bgr, panel_id, img_path.stem)
                if hole_dict and hole_dict.get("confidence", 0) >= _MIN_CONFIDENCE:
                    holes.append(hole_dict)
            except Exception as exc:
                log.debug("  Panel %s failed: %s", panel_id, exc)

        return holes

    # ── Panel splitting ───────────────────────────────────────────────────────

    def _split_panels(self, bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Detect whether the image is a grid of hole diagrams.

        Looks for thick horizontal/vertical separators (white or dark lines)
        separating panels. Returns list of (label, panel_image) tuples.
        Falls back to [] if no grid detected (caller treats whole image as one panel).
        """
        h, w = bgr.shape[:2]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Detect strong horizontal lines as panel separators
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 4, 1))
        horiz_mask   = cv2.morphologyEx(
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 5),
            cv2.MORPH_OPEN, horiz_kernel,
        )
        row_sums = horiz_mask.sum(axis=1)
        separator_rows = np.where(row_sums > w * 0.6)[0]

        if len(separator_rows) < 2:
            return []

        # Group adjacent separator rows into bands
        bands   = _group_indices(separator_rows, gap=5)
        cuts_y  = [int(np.mean(b)) for b in bands]

        if len(cuts_y) < 1:
            return []

        # Build panel list from row cuts
        row_cuts = [0] + cuts_y + [h]
        panels: List[Tuple[str, np.ndarray]] = []
        for i in range(len(row_cuts) - 1):
            y0, y1 = row_cuts[i], row_cuts[i + 1]
            if (y1 - y0) < h * 0.08:   # too thin — skip separator band
                continue
            panels.append((f"panel_{i}", bgr[y0:y1, :]))

        return panels if len(panels) >= 2 else []

    # ── Hole extraction ───────────────────────────────────────────────────────

    def _extract_hole(
        self,
        bgr: np.ndarray,
        panel_id: str,
        source_stem: str,
    ) -> Optional[Dict]:
        """
        Extract all features from a single-hole panel image.

        Returns a dict with:
            hole_number, fairways, greens, bunkers, water, tees,
            routing_line, confidence
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        h, w = bgr.shape[:2]

        # ── 1. Feature masks ─────────────────────────────────────────────────
        masks: Dict[str, np.ndarray] = {}
        for feat, ranges in _HSV_RANGES.items():
            combined = np.zeros(bgr.shape[:2], dtype=np.uint8)
            for r in ranges:
                lo  = np.array(r["lo"], dtype=np.uint8)
                hi  = np.array(r["hi"], dtype=np.uint8)
                combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
            masks[feat] = _apply_morph(combined, erode_k=2, dilate_k=3, close_k=6)

        # ── 2. Contour extraction ────────────────────────────────────────────
        fairways = _contours_to_polygons(masks["fairway"], bgr,
                                         min_area=_AREA_MIN_PX["fairway"],
                                         feat_type="fairway")
        greens   = _contours_to_polygons(masks["green"],   bgr,
                                         min_area=_AREA_MIN_PX["green"],
                                         feat_type="green")
        bunkers  = _contours_to_polygons(masks["bunker"],  bgr,
                                         min_area=_AREA_MIN_PX["bunker"],
                                         feat_type="bunker")
        water    = _contours_to_polygons(masks["water"],   bgr,
                                         min_area=_AREA_MIN_PX["water"],
                                         feat_type="water")
        tees     = _detect_tees(masks["tee"], masks["fairway"], bgr)

        # ── 3. Classify greens vs fairways by circularity ────────────────────
        pure_greens, extra_fairways = _classify_green_fairway(
            greens, _GREEN_CIRCULARITY_MIN
        )
        fairways.extend(extra_fairways)

        # ── 4. OCR for hole number ───────────────────────────────────────────
        hole_number = _ocr_hole_number(bgr)

        # ── 5. Confidence score ──────────────────────────────────────────────
        confidence = _compute_confidence(fairways, pure_greens, tees, hole_number)

        if confidence < _MIN_CONFIDENCE:
            log.debug("  Panel %s confidence %.2f < threshold — skipped", panel_id, confidence)
            return None

        # ── 6. Build routing line (tee → fairway centroid → green) ───────────
        routing = _build_routing_line(tees, fairways, pure_greens, w, h)

        # ── 7. Debug overlay ─────────────────────────────────────────────────
        if self.debug:
            tag = f"h{hole_number}" if hole_number else panel_id
            debug_path = self.output_dir / f"yardage_debug_{source_stem}_{tag}.png"
            _write_debug_overlay(bgr, fairways, pure_greens, bunkers, water, tees,
                                 routing, hole_number, debug_path)

        return {
            "hole_number": hole_number,
            "fairways":    fairways,
            "greens":      pure_greens,
            "bunkers":     bunkers,
            "water":       water,
            "tees":        tees,
            "routing":     routing,
            "confidence":  round(confidence, 3),
            "source":      source_stem,
            "panel":       panel_id,
        }

    # ── GeoJSON writer ────────────────────────────────────────────────────────

    def _write_geojsons(self, holes: List[Dict]) -> None:
        """Write per-feature-type GeoJSON files (pixel-space coordinates)."""
        feature_map: Dict[str, List] = {
            "fairways": [],
            "greens":   [],
            "bunkers":  [],
            "water":    [],
            "holes":    [],
        }

        for hole in holes:
            hn = hole.get("hole_number")
            props_base = {"hole_number": hn, "confidence": hole["confidence"],
                          "source": hole["source"]}

            for feat_type in ("fairways", "greens", "bunkers", "water"):
                for poly in hole.get(feat_type, []):
                    feature_map[feat_type].append(_polygon_to_geojson_feature(
                        poly["contour"], {**props_base, "type": feat_type}
                    ))

            if hole.get("routing"):
                feature_map["holes"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": hole["routing"],
                    },
                    "properties": props_base,
                })

        for feat_type, features in feature_map.items():
            out_path = self.output_dir / f"yardage_{feat_type}.geojson"
            fc = {"type": "FeatureCollection", "features": features}
            out_path.write_text(json.dumps(fc, indent=2))
            log.info("  Written: %s (%d features)", out_path.name, len(features))


# ─── Utility functions ────────────────────────────────────────────────────────

def _normalise_image(bgr: np.ndarray, target_long_edge: int = 2400) -> np.ndarray:
    """Upscale small images; never downscale below target_long_edge."""
    h, w = bgr.shape[:2]
    long_edge = max(h, w)
    if long_edge < target_long_edge:
        scale = target_long_edge / long_edge
        new_w = int(w * scale)
        new_h = int(h * scale)
        bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    return bgr


def _apply_morph(
    mask: np.ndarray,
    erode_k: int = 2,
    dilate_k: int = 4,
    close_k: int = 8,
) -> np.ndarray:
    """Erode → Dilate → Morphological Close to clean binary mask."""
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
    out = cv2.erode(mask,  ke, iterations=1)
    out = cv2.dilate(out,  kd, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kc)
    return out


def _contours_to_polygons(
    mask: np.ndarray,
    bgr: np.ndarray,
    min_area: int,
    feat_type: str,
) -> List[Dict]:
    """
    Extract contours from mask, filter by area, and return polygon dicts.

    Each dict has:
        contour   : np.ndarray  (Nx2 pixel coords)
        area_px   : float
        centroid  : (cx, cy) float pixels
        bbox      : (x, y, w, h)
        circularity: float
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[Dict] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        perim = cv2.arcLength(c, True)
        circ  = (4 * np.pi * area / (perim ** 2)) if perim > 0 else 0.0
        x, y, w, h = cv2.boundingRect(c)
        polys.append({
            "contour":     c.reshape(-1, 2),
            "area_px":     float(area),
            "centroid":    (float(cx), float(cy)),
            "bbox":        (x, y, w, h),
            "circularity": float(circ),
            "feat_type":   feat_type,
        })
    # Largest-first
    polys.sort(key=lambda p: p["area_px"], reverse=True)
    return polys


def _classify_green_fairway(
    green_polys: List[Dict],
    circ_threshold: float,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Re-classify green-coloured regions:
      - high circularity + smaller area → putting green
      - low circularity or large area   → fairway segment
    """
    pure_greens: List[Dict] = []
    extra_fairways: List[Dict] = []
    for p in green_polys:
        if p["circularity"] >= circ_threshold:
            p["feat_type"] = "green"
            pure_greens.append(p)
        else:
            p["feat_type"] = "fairway"
            extra_fairways.append(p)
    return pure_greens, extra_fairways


def _detect_tees(
    tee_mask: np.ndarray,
    fairway_mask: np.ndarray,
    bgr: np.ndarray,
) -> List[Dict]:
    """
    Detect tee boxes from tee_mask using approxPolyDP rectangularity test.

    A tee box is a small-ish rectangle, distinctly coloured (red/blue/white)
    at the start of a fairway corridor.
    """
    cnts, _ = cv2.findContours(tee_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tees: List[Dict] = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < _AREA_MIN_PX["tee"]:
            continue
        arc  = float(cv2.arcLength(c, closed=True))
        if arc < 1:
            continue
        eps  = 0.04 * arc
        poly = cv2.approxPolyDP(c, eps, closed=True)
        if len(poly) < 4 or len(poly) > 8:
            continue
        rect     = cv2.minAreaRect(c)
        w_r, h_r = rect[1]
        if min(w_r, h_r) < 1:
            continue
        aspect = max(w_r, h_r) / min(w_r, h_r)
        if not (1.0 <= aspect <= 10.0):
            continue
        M = cv2.moments(c)
        cx = M["m10"] / M["m00"] if M["m00"] > 0 else 0.0
        cy = M["m01"] / M["m00"] if M["m00"] > 0 else 0.0
        x, y, w, h = cv2.boundingRect(c)
        tees.append({
            "contour":  c.reshape(-1, 2),
            "area_px":  float(area),
            "centroid": (float(cx), float(cy)),
            "bbox":     (x, y, w, h),
            "feat_type": "tee",
        })
    tees.sort(key=lambda t: t["area_px"], reverse=True)
    return tees[:4]   # at most 4 tee boxes (championship/medal/ladies/junior)


def _build_routing_line(
    tees:     List[Dict],
    fairways: List[Dict],
    greens:   List[Dict],
    img_w:    int,
    img_h:    int,
) -> List[List[float]]:
    """
    Build a routing LineString: tee_centroid → fairway_centroids → green_centroid.

    Returns list of [x, y] pixel coordinates (image-space, normalised 0-1).
    Falls back gracefully when features are missing.
    """
    waypoints: List[Tuple[float, float]] = []

    # Tee: use largest detected tee, else synthesise from image top region
    if tees:
        waypoints.append(tees[0]["centroid"])
    else:
        # No tee detected — assume top-centre of image
        waypoints.append((img_w / 2, img_h * 0.1))

    # Fairway centroids ordered by proximity to previous waypoint
    remaining = list(fairways)
    while remaining:
        prev = waypoints[-1]
        remaining.sort(key=lambda p: _dist(p["centroid"], prev))
        nearest = remaining.pop(0)
        # Only add if significantly farther than already-added point
        if len(waypoints) < 2 or _dist(nearest["centroid"], waypoints[-1]) > 30:
            waypoints.append(nearest["centroid"])

    # Green: use largest green, else synthesise from image bottom region
    if greens:
        waypoints.append(greens[0]["centroid"])
    else:
        waypoints.append((img_w / 2, img_h * 0.9))

    # Normalise to 0–1 range
    return [[round(x / img_w, 4), round(y / img_h, 4)] for x, y in waypoints]


def _ocr_hole_number(bgr: np.ndarray) -> Optional[int]:
    """
    OCR the hole number from the image.

    Tries pytesseract on the top and bottom bands of the image.
    Returns an integer 1–18 or None if not found / OCR not available.
    """
    try:
        import pytesseract
        from PIL import Image

        _tesseract_cmd = getattr(config, "TESSERACT_CMD", "")
        if _tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd

        h, w = bgr.shape[:2]
        # Scan top 20% and bottom 20% for hole number text
        regions = [
            bgr[:int(h * 0.22), :],
            bgr[int(h * 0.78):, :],
        ]
        for region in regions:
            for threshold in [
                lambda g: cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                lambda g: cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 11, 2),
            ]:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                try:
                    thresh = threshold(gray)
                except Exception:
                    continue
                pil_img = Image.fromarray(thresh)
                data = pytesseract.image_to_data(
                    pil_img,
                    config="--psm 7 -c tessedit_char_whitelist=0123456789 "
                           "outputbase digits",
                    output_type=pytesseract.Output.DICT,
                )
                for txt, conf in zip(data["text"], data["conf"]):
                    txt = str(txt).strip()
                    try:
                        conf_val = int(conf)
                    except (ValueError, TypeError):
                        conf_val = 0
                    if conf_val >= _OCR_CONF_MIN and txt.isdigit():
                        n = int(txt)
                        if 1 <= n <= 18:
                            return n

    except ImportError:
        log.debug("pytesseract not installed — hole number OCR skipped")
    except Exception as exc:
        log.debug("OCR error: %s", exc)

    return None


def _compute_confidence(
    fairways: List[Dict],
    greens:   List[Dict],
    tees:     List[Dict],
    hole_number: Optional[int],
) -> float:
    """
    Heuristic confidence score in [0, 1] based on detected features.

    Partial scoring:
        +0.30  at least one fairway detected
        +0.25  at least one green detected
        +0.15  at least one tee detected
        +0.20  hole number successfully OCR'd
        +0.10  multiple fairway segments (realistic routing)
    """
    score = 0.0
    if fairways:
        score += 0.30
    if greens:
        score += 0.25
    if tees:
        score += 0.15
    if hole_number is not None:
        score += 0.20
    if len(fairways) >= 2:
        score += 0.10
    return min(score, 1.0)


def _deduplicate_holes(holes: List[Dict]) -> List[Dict]:
    """
    When multiple images contain the same hole number, keep highest-confidence entry.
    Holes with no OCR'd number are kept as-is.
    """
    best: Dict[Optional[int], Dict] = {}
    no_number: List[Dict] = []
    for h in holes:
        hn = h.get("hole_number")
        if hn is None:
            no_number.append(h)
        elif hn not in best or h["confidence"] > best[hn]["confidence"]:
            best[hn] = h
    return list(best.values()) + no_number


def _holes_to_featurecollection(holes: List[Dict]) -> Dict:
    """Convert list of hole dicts to a merged GeoJSON FeatureCollection."""
    features = []
    for hole in holes:
        hn = hole.get("hole_number")
        for feat_type in ("fairways", "greens", "bunkers", "water"):
            for poly in hole.get(feat_type, []):
                features.append(_polygon_to_geojson_feature(
                    poly["contour"],
                    {"hole_number": hn, "type": feat_type, "confidence": hole["confidence"]},
                ))
        if hole.get("routing"):
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": hole["routing"]},
                "properties": {"hole_number": hn, "type": "routing",
                               "confidence": hole["confidence"]},
            })
    return {"type": "FeatureCollection", "features": features}


def _polygon_to_geojson_feature(contour: np.ndarray, properties: Dict) -> Dict:
    """Convert an Nx2 pixel contour to a GeoJSON Polygon Feature."""
    coords = contour.tolist()
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])   # close ring
    return {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": properties,
    }


def _empty_fc() -> Dict:
    return {"type": "FeatureCollection", "features": []}


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _group_indices(indices: np.ndarray, gap: int = 5) -> List[List[int]]:
    """Group consecutive indices into bands separated by gaps > `gap`."""
    if len(indices) == 0:
        return []
    groups: List[List[int]] = [[int(indices[0])]]
    for idx in indices[1:]:
        if int(idx) - groups[-1][-1] <= gap:
            groups[-1].append(int(idx))
        else:
            groups.append([int(idx)])
    return groups


# ─── Debug overlay ────────────────────────────────────────────────────────────

_DEBUG_COLOURS = {
    "fairway": (34,  139, 34),   # ForestGreen (BGR)
    "green":   (0,   200, 0),    # bright green
    "bunker":  (0,   165, 255),  # orange
    "water":   (255, 100, 0),    # blue
    "tee":     (0,   0,   220),  # red
    "routing": (50,  50,  255),  # bold red line
}

_LEGEND_LABELS = [
    ("fairway", "Fairway"),
    ("green",   "Green"),
    ("bunker",  "Bunker"),
    ("water",   "Water"),
    ("tee",     "Tee box"),
]


def _write_debug_overlay(
    bgr:         np.ndarray,
    fairways:    List[Dict],
    greens:      List[Dict],
    bunkers:     List[Dict],
    water:       List[Dict],
    tees:        List[Dict],
    routing:     List[List[float]],
    hole_number: Optional[int],
    output_path: Path,
) -> None:
    """Write a debug overlay PNG for one hole panel."""
    overlay = bgr.copy().astype(np.float32)
    h, w    = bgr.shape[:2]

    feature_groups = [
        (fairways, "fairway"),
        (greens,   "green"),
        (bunkers,  "bunker"),
        (water,    "water"),
        (tees,     "tee"),
    ]

    # Semi-transparent filled polygons
    for polys, feat_type in feature_groups:
        colour = _DEBUG_COLOURS[feat_type]
        for poly in polys:
            cnt = poly["contour"].reshape(-1, 1, 2).astype(np.int32)
            mask_layer = np.zeros_like(bgr, dtype=np.uint8)
            cv2.fillPoly(mask_layer, [cnt], colour)
            overlay = cv2.addWeighted(overlay, 1.0,
                                      mask_layer.astype(np.float32), 0.4, 0)

    canvas = np.clip(overlay, 0, 255).astype(np.uint8)

    # Routing line with arrowheads
    if routing:
        pts = [(int(x * w), int(y * h)) for x, y in routing]
        for i in range(len(pts) - 1):
            cv2.arrowedLine(canvas, pts[i], pts[i + 1],
                            _DEBUG_COLOURS["routing"], 3, tipLength=0.04)

    # Hole number label
    if hole_number is not None:
        label = f"Hole {hole_number}"
        cv2.putText(canvas, label, (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 5)
        cv2.putText(canvas, label, (20, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 2)

    # Legend (bottom-left)
    legend_y = h - (len(_LEGEND_LABELS) * 32 + 20)
    for feat_type, label_text in _LEGEND_LABELS:
        colour = _DEBUG_COLOURS[feat_type]
        cv2.rectangle(canvas, (12, legend_y - 18), (32, legend_y + 4), colour, -1)
        cv2.putText(canvas, label_text, (38, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3)
        cv2.putText(canvas, label_text, (38, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
        legend_y += 32

    cv2.imwrite(str(output_path), canvas)
    log.debug("  Debug overlay: %s", output_path)


# ─── Convenience entry point (standalone execution) ───────────────────────────

def extract_yardage_books(
    assets_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    debug: bool = False,
) -> Dict:
    """
    Top-level function for pipeline integration.

    Parameters
    ----------
    assets_dir : Path, optional
        Defaults to config.ASSETS_DIR.
    output_dir : Path, optional
        Defaults to config.OUTPUT_DIR / "yardage_ai".
    debug : bool
        Write per-hole debug PNGs.

    Returns
    -------
    dict
        {"holes": [...], "geojson": FeatureCollection}
    """
    if assets_dir is None:
        assets_dir = Path(getattr(config, "ASSETS_DIR", "assets"))
    if output_dir is None:
        output_dir = Path(getattr(config, "OUTPUT_DIR", "output")) / "yardage_ai"

    extractor = YardageBookExtractor(assets_dir, output_dir, debug=debug)
    return extractor.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="Yardage book AI extractor")
    parser.add_argument("--assets", default=str(_ASSETS_DIR),
                        help="Assets directory (contains yardage_books/)")
    parser.add_argument("--output", default="output/yardage_ai",
                        help="Output directory for GeoJSON files")
    parser.add_argument("--debug", action="store_true",
                        help="Write per-hole debug overlay PNGs")
    args = parser.parse_args()

    result = extract_yardage_books(
        assets_dir=Path(args.assets),
        output_dir=Path(args.output),
        debug=args.debug,
    )
    print(f"Extracted {len(result['holes'])} hole(s).")
    print(f"GeoJSON features: {len(result['geojson']['features'])}")

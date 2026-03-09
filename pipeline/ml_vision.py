"""
ml_vision.py — Optional ML-based vision refinement module.

This module provides an ML post-processing step that can refine the
colour-threshold detections from vision_extract.py.

DISABLED BY DEFAULT: config.ENABLE_ML_VISION = False

When disabled the module is never imported and the pipeline behaves
exactly as before. Set ENABLE_ML_VISION = True in config.py and install
the optional dependencies to enable.

When enabled:
  - Loads detections from vision GeoJSON files
  - Passes each detection through a lightweight classifier/refiner
  - Updates confidence scores and may remove false positives
  - Saves refined GeoJSON back to the same paths

Optional dependencies (not installed by default):
  torch, torchvision, transformers, segment-anything

Current implementation: rule-based refinement placeholder that
demonstrates the interface. Replace the _ml_refine_features() body
with a real model when available.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)

ENABLED = getattr(config, "ENABLE_ML_VISION", False)


# ─── Public API ───────────────────────────────────────────────────────────────

def refine_vision_detections(
    output_dir: Path,
    satellite_mosaic_path: Optional[Path] = None,
) -> dict:
    """
    Refine vision detections using ML (if ENABLE_ML_VISION = True).

    Args:
        output_dir:             Directory containing vision_*.geojson files.
        satellite_mosaic_path:  Path to satellite_mosaic.jpg for context.

    Returns:
        Summary dict with per-type refinement statistics, or empty dict
        if ML is disabled.
    """
    if not ENABLED:
        log.debug("ML vision refinement is disabled (ENABLE_ML_VISION = False)")
        return {}

    output_dir = Path(output_dir)
    stats: Dict[str, dict] = {}

    feature_types = ["bunker", "green", "fairway", "water", "trees", "rough"]

    for feat_type in feature_types:
        vision_path = output_dir / f"vision_{feat_type}s.geojson"
        if not vision_path.exists():
            continue

        try:
            features = _load_geojson(vision_path)
            before   = len(features)
            refined  = _ml_refine_features(features, feat_type, satellite_mosaic_path)
            after    = len(refined)

            _write_geojson(refined, vision_path)

            stats[feat_type] = {
                "before": before,
                "after":  after,
                "removed": before - after,
            }
            if before != after:
                log.info(
                    f"ML vision refinement [{feat_type}]: "
                    f"{before} → {after} features ({before - after} removed)"
                )
        except Exception as e:
            log.warning(f"ML refinement failed for {feat_type}: {e}")

    return stats


# ─── ML refinement (stub — replace with real model) ──────────────────────────

def _ml_refine_features(
    features: List[dict],
    feature_type: str,
    mosaic_path: Optional[Path],
) -> List[dict]:
    """
    Refine a list of GeoJSON features using ML.

    Current implementation: rule-based confidence thresholding.
    Replace this function body with a real ML model when available.

    Returns filtered/refined feature list.
    """
    # Minimum confidence for each feature type
    min_confidence = {
        "bunker":  0.50,
        "green":   0.55,
        "fairway": 0.45,
        "water":   0.40,
        "trees":   0.35,
        "rough":   0.30,
    }
    threshold = min_confidence.get(feature_type, 0.40)

    refined = []
    for f in features:
        conf = f.get("properties", {}).get("confidence", 0.5)
        if conf >= threshold:
            # Optionally boost confidence for high-quality detections
            if conf >= 0.7:
                f["properties"]["ml_verified"] = True
            refined.append(f)

    return refined


# ─── Utilities ────────────────────────────────────────────────────────────────

def _load_geojson(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("features", [])
    except Exception:
        return []


def _write_geojson(features: List[dict], out_path: Path) -> None:
    out_path.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )

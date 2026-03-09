"""
sam_segment.py — Optional Segment Anything Model (SAM) integration.

UPGRADE 2 (optional): High-accuracy satellite segmentation via SAM.

When ENABLE_SAM = True in config.py and segment-anything + PyTorch are
installed, this module enhances feature detection by running SAM's automatic
mask generator over the satellite mosaic and classifying each segment into
golf feature types using colour + shape heuristics.

When ENABLE_SAM = False (default) or dependencies are missing, every public
function in this module returns an empty dict and the pipeline proceeds with
standard HSV-based detection unchanged.

Installation (optional heavy dependencies):
    pip install torch torchvision
    pip install git+https://github.com/facebookresearch/segment-anything.git
    # Download ViT-B checkpoint (~375 MB):
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Config keys (config.py):
    ENABLE_SAM      = True                      # opt-in
    SAM_CHECKPOINT  = "sam_vit_b_01ec64.pth"   # path to checkpoint file
    SAM_MODEL_TYPE  = "vit_b"                   # vit_h | vit_l | vit_b

Segment classification heuristics:
    fairway — elongated, green hue, low texture (uniform)
    green   — compact oval, green hue, very uniform (low std)
    bunker  — bright, sandy hue, low-medium saturation
    water   — blue/grey hue or very dark value
    tee     — small, rectangular, green hue
    rough   — medium-texture green
    trees   — dark green, high texture

All classifications are tentative confidence boosts only.  The existing
HSV detection pipeline remains the authoritative source of feature polygons.
SAM segments are merged with HSV detections when they provide additional
coverage (IoU < threshold).
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import importlib.util as _ilu
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.py")
if "config" not in sys.modules or not hasattr(sys.modules["config"], "OVERPASS_URL"):
    _spec = _ilu.spec_from_file_location("config", _CONFIG_PATH)
    _mod  = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    sys.modules["config"] = _mod
import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger(__name__)


# ─── Availability check ───────────────────────────────────────────────────────

def is_sam_available() -> bool:
    """Return True when SAM is enabled in config and all deps are installed."""
    if not getattr(config, "ENABLE_SAM", False):
        return False
    try:
        import torch  # noqa: F401
        from segment_anything import sam_model_registry  # noqa: F401
        checkpoint = Path(getattr(config, "SAM_CHECKPOINT", "sam_vit_b_01ec64.pth"))
        if not checkpoint.exists():
            log.warning(
                f"SAM enabled but checkpoint not found: {checkpoint}. "
                f"Download from https://dl.fbaipublicfiles.com/segment_anything/"
            )
            return False
        return True
    except ImportError as e:
        log.warning(
            f"SAM dependencies not installed ({e}). "
            f"Install with: pip install torch torchvision segment-anything"
        )
        return False


# ─── Public API ───────────────────────────────────────────────────────────────

def run_sam_segmentation(
    img_rgb,
    output_dir: Path,
) -> Dict[str, list]:
    """
    Run SAM automatic mask generation and classify segments by golf feature type.

    Args:
        img_rgb:    H×W×3 uint8 RGB image array (satellite mosaic)
        output_dir: Directory to write sam_segments.json cache

    Returns:
        Dict mapping feature_type → list of SAM mask dicts.
        Returns {} when SAM is disabled or unavailable.
    """
    if not is_sam_available():
        return {}

    try:
        import numpy as np
        import torch
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        checkpoint  = Path(getattr(config, "SAM_CHECKPOINT", "sam_vit_b_01ec64.pth"))
        model_type  = getattr(config, "SAM_MODEL_TYPE", "vit_b")
        device      = "cuda" if torch.cuda.is_available() else "cpu"

        log.info(f"SAM: loading {model_type} checkpoint on {device} ...")
        sam       = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(device=device)
        generator = SamAutomaticMaskGenerator(
            sam,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            min_mask_region_area=100,
            points_per_side=32,
        )

        log.info("SAM: running automatic segmentation ...")
        masks = generator.generate(img_rgb)
        log.info(f"SAM: {len(masks)} segments generated")

        classified = _classify_sam_masks(masks, img_rgb)

        # Write cache summary
        summary = {
            "total_segments": len(masks),
            "classified":     {k: len(v) for k, v in classified.items()},
        }
        (Path(output_dir) / "sam_segments.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        log.info(f"SAM classification: {summary['classified']}")
        return classified

    except Exception as e:
        log.warning(f"SAM segmentation failed: {e}")
        return {}


# ─── Segment classification ───────────────────────────────────────────────────

def _classify_sam_masks(masks: list, img_rgb) -> Dict[str, list]:
    """
    Classify SAM segments into golf feature types using colour/shape heuristics.

    Each SAM mask dict has: "segmentation" (H×W bool), "area", "bbox",
    "predicted_iou", "stability_score".

    Returns dict: feature_type → [mask_dict, ...]
    """
    import numpy as np

    try:
        import cv2
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    except ImportError:
        return {}

    classified: Dict[str, list] = {
        "fairway": [], "green":  [], "bunker": [],
        "water":   [], "tee":    [], "rough":  [], "trees": [],
    }

    for mask_dict in masks:
        seg = mask_dict.get("segmentation")
        if seg is None or not seg.any():
            continue

        area_px = int(seg.sum())
        bx, by, bw, bh = mask_dict.get("bbox", [0, 0, 1, 1])
        aspect = max(bw, bh) / max(min(bw, bh), 1)

        # Mean HSV colour inside segment
        h_mean = float(np.mean(hsv[:, :, 0][seg]))
        s_mean = float(np.mean(hsv[:, :, 1][seg]))
        v_mean = float(np.mean(hsv[:, :, 2][seg]))
        v_std  = float(np.std(hsv[:, :, 2][seg]))   # texture proxy

        feat = None

        # Water: blue/grey hue or very dark
        if (90 <= h_mean <= 140 and v_mean < 130) or v_mean < 45:
            if area_px >= 150:
                feat = "water"

        # Bunker: bright, sandy hue, low saturation
        elif 14 <= h_mean <= 45 and s_mean < 120 and v_mean > 155:
            if 50 <= area_px <= 3000:
                feat = "bunker"

        # Green: small, compact, very uniform green
        elif 35 <= h_mean <= 80 and s_mean > 40 and v_std < 12:
            if 60 <= area_px <= 700 and aspect < 2.2:
                feat = "green"

        # Tee: small, rectangular, green
        elif 35 <= h_mean <= 80 and s_mean > 30 and aspect < 3.0:
            if 25 <= area_px <= 200:
                feat = "tee"

        # Fairway: large elongated green, low texture
        elif 35 <= h_mean <= 80 and s_mean > 30 and v_std < 25:
            if area_px >= 400 and aspect >= 1.5:
                feat = "fairway"

        # Trees: dark green, high texture
        elif 30 <= h_mean <= 80 and v_mean < 100 and v_std > 20:
            if area_px >= 100:
                feat = "trees"

        # Rough: medium-texture green
        elif 28 <= h_mean <= 85 and s_mean > 20:
            if area_px >= 200 and 12 <= v_std <= 45:
                feat = "rough"

        if feat:
            classified[feat].append(mask_dict)

    return classified

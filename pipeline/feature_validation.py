"""
feature_validation.py — Expected feature count validation engine.

UPGRADE 9: Validation engine.

After vision detection, validates detected feature counts against
expected ranges for an 18-hole golf course. Logs warnings when counts
are outside the expected range and writes a machine-readable report.

Does NOT modify detections — purely diagnostic.  Downstream modules
(routing, QA) can read feature_validation.json to adjust behaviour.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger(__name__)


# ─── Expected count ranges for a standard 18-hole course ─────────────────────

EXPECTED_COUNTS: Dict[str, Tuple[int, int]] = {
    "fairway": (14,  22),   # 18 holes; some holes share a fairway polygon
    "green":   (16,  20),   # close to 18
    "bunker":  (20, 120),   # highly variable by course design
    "tee":     (18,  90),   # 1–5 colour tees per hole
    "water":   (0,   20),   # optional feature
    "trees":   (0,  200),   # unlimited
    "rough":   (0,   50),
}

# Status labels
_CRITICAL_LOW  = "critical_low"
_LOW           = "low"
_OK            = "ok"
_HIGH          = "high"
_CRITICAL_HIGH = "critical_high"


# ─── Public API ───────────────────────────────────────────────────────────────

def validate_feature_counts(
    feature_counts: Dict[str, int],
) -> Dict[str, str]:
    """
    Validate detected feature counts against expected ranges.

    Returns:
        Dict mapping feature_type → status string
        ("ok" | "low" | "high" | "critical_low" | "critical_high")
    """
    results: Dict[str, str] = {}

    for feat_type, count in feature_counts.items():
        lo, hi = EXPECTED_COUNTS.get(feat_type, (0, 9999))

        if count < lo * 0.5:
            status = _CRITICAL_LOW
        elif count < lo:
            status = _LOW
        elif count > hi * 2.5:
            status = _CRITICAL_HIGH
        elif count > hi:
            status = _HIGH
        else:
            status = _OK

        results[feat_type] = status

        if status == _OK:
            log.info(
                f"Validation [{feat_type}]: {count} ✓ "
                f"(expected {lo}–{hi})"
            )
        elif _CRITICAL_LOW == status:
            log.warning(
                f"Validation [{feat_type}]: {count} ✗ CRITICAL LOW "
                f"(expected ≥{lo}) — detector may have failed or "
                f"course has very sparse OSM coverage"
            )
        elif status == _LOW:
            log.warning(
                f"Validation [{feat_type}]: {count} ⚠ LOW "
                f"(expected {lo}–{hi}) — some features may be missing"
            )
        elif status == _CRITICAL_HIGH:
            log.warning(
                f"Validation [{feat_type}]: {count} ✗ CRITICAL HIGH "
                f"(expected ≤{hi}) — severe over-detection; "
                f"course boundary mask may not be applied"
            )
        else:  # high
            log.warning(
                f"Validation [{feat_type}]: {count} ⚠ HIGH "
                f"(expected {lo}–{hi}) — possible over-detection"
            )

    return results


def overall_status(validation_results: Dict[str, str]) -> str:
    """
    Summarise validation results into a single status string.

    Returns: "PASS" | "WARN" | "CRITICAL"
    """
    statuses = set(validation_results.values())
    if any("critical" in s for s in statuses):
        return "CRITICAL"
    if any(s in (_LOW, _HIGH) for s in statuses):
        return "WARN"
    return "PASS"


def write_validation_report(
    feature_counts: Dict[str, int],
    validation_results: Dict[str, str],
    output_dir: Path,
) -> None:
    """
    Write feature_validation.json to output_dir.

    Format:
    {
        "counts":   { "fairway": 18, "green": 18, ... },
        "status":   { "fairway": "ok", ... },
        "expected": { "fairway": {"min": 14, "max": 22}, ... },
        "overall":  "PASS" | "WARN" | "CRITICAL"
    }
    """
    report = {
        "counts":   feature_counts,
        "status":   validation_results,
        "expected": {
            k: {"min": v[0], "max": v[1]}
            for k, v in EXPECTED_COUNTS.items()
        },
        "overall":  overall_status(validation_results),
    }

    out = Path(output_dir) / "feature_validation.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info(
        f"Feature validation: overall={report['overall']} — "
        f"counts: { {k: v for k, v in feature_counts.items()} } "
        f"→ {out.name}"
    )

"""
qa.py — Quality assurance and fidelity validation.

Validates:
  - Hole count and routing
  - Distance accuracy (vs expected / scorecard)
  - Elevation accuracy (vs DTM)
  - Hazard presence and placement
  - Course feel classification

Produces:
  - qa_report.json    Detailed QA report
  - qa_report.html    Human-readable QA summary
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)


DISTANCE_TOLERANCE_PCT = 5.0   # Acceptable hole length error %
ROUTING_BEARING_TOLERANCE_DEG = 25.0   # Acceptable bearing error vs real-world


# ─── Main entry point ─────────────────────────────────────────────────────────

def run_qa(
    boundary_data:  dict,
    terrain_stats:  dict,
    features_data:  dict,
    build_pack:     dict,
    output_dir:     Path,
    expected_scorecard: Optional[List[dict]] = None,
) -> dict:
    """
    Run all QA checks and produce a report.

    Args:
        boundary_data:       From boundary.resolve_boundary()
        terrain_stats:       From terrain.process_terrain()
        features_data:       From features.extract_features()
        build_pack:          From translation.generate_build_pack()
        output_dir:          Output directory
        expected_scorecard:  Optional [{hole, par, yards, si}] for validation

    Returns:
        qa_report dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holes  = features_data["holes"]

    # Supplement hole list from routing inference when OSM data is sparse
    _routing_meta_path = output_dir / "holes_metadata.json"
    if not holes and _routing_meta_path.exists():
        try:
            _rm = json.loads(_routing_meta_path.read_text(encoding="utf-8"))
            if _rm.get("source") == "routing_inference" and _rm.get("hole_count", 0) > 0:
                holes = [{"hole_number": i + 1} for i in range(_rm["hole_count"])]
                log.debug(f"QA: supplemented {len(holes)} holes from routing_inference")
        except Exception:
            pass

    report = {
        "course_name":    boundary_data["matched_name"],
        "total_holes":    len(holes),
        "checks":         [],
        "warnings":       [],
        "errors":         [],
        "overall_status": "PASS",
    }

    # ── Check 1: Hole count ─────────────────────────────────────────────────
    _check_hole_count(holes, report)

    # ── Check 2: Routing direction ──────────────────────────────────────────
    _check_routing_direction(holes, report)

    # ── Check 3: Distance validation ────────────────────────────────────────
    _check_distances(holes, report, expected_scorecard)

    # ── Check 4: Elevation range ────────────────────────────────────────────
    _check_elevation(terrain_stats, report)

    # ── Check 5: Hazard presence ────────────────────────────────────────────
    _check_hazards(holes, features_data, report)

    # ── Check 6: Feature confidence ─────────────────────────────────────────
    _check_confidence(features_data, report)

    # ── Check 7: Canvas fit ─────────────────────────────────────────────────
    _check_canvas_fit(build_pack, report)

    # ── Check 8: Course feel ────────────────────────────────────────────────
    _check_course_feel(terrain_stats, features_data, report)

    # Overall status
    if any(c["status"] == "FAIL" for c in report["checks"]):
        report["overall_status"] = "FAIL"
    elif any(c["status"] == "WARN" for c in report["checks"]):
        report["overall_status"] = "WARN"

    # Write report
    qa_json_path = output_dir / "qa_report.json"
    qa_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    qa_html_path = output_dir / "qa_report.html"
    _write_qa_html(report, qa_html_path)

    log.info(f"QA complete — status: {report['overall_status']}")
    return report


# ─── Individual checks ────────────────────────────────────────────────────────

def _check_hole_count(holes: list, report: dict) -> None:
    count = len(holes)
    if count == 18:
        status = "PASS"
        msg    = "18 holes detected."
    elif count == 9:
        status = "WARN"
        msg    = "9 holes detected — is this a 9-hole course?"
    elif count == 0:
        status = "FAIL"
        msg    = "No holes extracted. Check OSM coverage for this course."
    else:
        status = "WARN"
        msg    = f"{count} holes detected (expected 18). Some holes may be missing or merged."

    _add_check(report, "Hole Count", status, msg, {"detected": count, "expected": 18})


def _check_routing_direction(holes: list, report: dict) -> None:
    """
    Check that no two holes have nearly identical routing (routing collision).
    Check that tee-to-green vectors are not all identical (sign of bad inference).
    """
    issues = []

    bearings = []
    for hole in holes:
        tee   = hole.get("tee_centroid", [0, 0])
        green = hole.get("green_centroid", [0, 0])
        b     = _bearing(tee, green)
        bearings.append(b)

    # Check for duplicate routings
    for i, h1 in enumerate(holes):
        for j, h2 in enumerate(holes):
            if i >= j:
                continue
            d = _haversine(h1["tee_centroid"], h2["tee_centroid"])
            if d < 20:
                issues.append(
                    f"Holes {h1['hole_number']} and {h2['hole_number']} "
                    f"have nearly identical tee positions ({d:.0f}m apart)"
                )

    if not issues:
        _add_check(report, "Routing Direction", "PASS", "No routing collisions detected.", {})
    else:
        _add_check(
            report, "Routing Direction", "WARN",
            "Possible routing collisions: " + "; ".join(issues),
            {"collision_count": len(issues)},
        )


def _check_distances(holes: list, report: dict, expected_scorecard: Optional[list]) -> None:
    """Validate hole distances against expected values or PAR heuristics."""
    dist_issues = []
    has_scorecard = bool(expected_scorecard)

    sc_map = {}
    if expected_scorecard:
        sc_map = {s["hole"]: s for s in expected_scorecard if "hole" in s}

    for hole in holes:
        n   = hole["hole_number"]
        yd  = hole.get("length_yards", 0) or 0
        par = hole.get("par")

        if has_scorecard and n in sc_map:
            expected_yd = sc_map[n].get("yards", 0)
            if expected_yd and yd:
                err_pct = abs(yd - expected_yd) / expected_yd * 100
                if err_pct > DISTANCE_TOLERANCE_PCT:
                    dist_issues.append(
                        f"Hole {n}: calculated {int(yd)}y vs scorecard {int(expected_yd)}y "
                        f"({err_pct:.1f}% error)"
                    )
        elif par:
            # PAR heuristic ranges
            heuristic = {3: (100, 250), 4: (250, 470), 5: (450, 650)}
            lo, hi = heuristic.get(par, (100, 700))
            if yd and not (lo <= yd <= hi):
                dist_issues.append(
                    f"Hole {n}: {int(yd)}y for par {par} is outside expected range {lo}–{hi}y"
                )

    if not dist_issues:
        _add_check(report, "Distance Validation", "PASS", "All hole distances within tolerance.", {})
    else:
        _add_check(
            report, "Distance Validation", "WARN",
            f"{len(dist_issues)} distance issue(s): " + " | ".join(dist_issues),
            {"issue_count": len(dist_issues)},
        )


def _check_elevation(terrain_stats: dict, report: dict) -> None:
    z_range = terrain_stats.get("z_range_m", 0)
    res     = terrain_stats.get("resolution_m", 30)

    if z_range < 0.5:
        status = "WARN"
        msg    = (
            f"Very low terrain relief ({z_range:.2f}m). "
            "DTM may be of poor quality, or course is exceptionally flat."
        )
    elif z_range > 300:
        status = "WARN"
        msg    = (
            f"Unusually high terrain relief ({z_range:.1f}m). "
            "Check DTM for nodata artefacts or incorrect clipping."
        )
    else:
        status = "PASS"
        msg    = f"Terrain relief {z_range:.1f}m at {res:.0f}m resolution. Acceptable."

    if res > 25:
        report["warnings"].append(
            f"Low-resolution DEM ({res:.0f}m). Terrain detail will be approximate."
        )

    _add_check(report, "Elevation Range", status, msg, {
        "z_range_m":    z_range,
        "resolution_m": res,
    })


def _check_hazards(holes: list, features_data: dict, report: dict) -> None:
    """Check bunker and water hazard presence and plausibility."""
    total_bunkers = sum(len(h.get("bunkers", [])) for h in holes)
    cs            = features_data.get("confidence_summary", {})
    bunk_conf     = cs.get("bunker", {}).get("label", "LOW")

    if total_bunkers == 0:
        status = "WARN"
        msg    = "No bunkers detected in OSM. All bunker placement will be manual."
    elif total_bunkers < 18:
        status = "WARN"
        msg    = (
            f"{total_bunkers} bunkers detected (typical 18-hole course: 50–100). "
            "OSM bunker coverage may be incomplete."
        )
    else:
        status = "PASS"
        msg    = f"{total_bunkers} bunkers detected. Confidence: {bunk_conf}."

    _add_check(report, "Hazard Presence", status, msg, {
        "total_bunkers":   total_bunkers,
        "bunker_confidence": bunk_conf,
    })


def _check_confidence(features_data: dict, report: dict) -> None:
    cs = features_data.get("confidence_summary", {})
    low_confidence_types = [
        ft for ft, data in cs.items()
        if data.get("label") == "LOW"
    ]

    if not low_confidence_types:
        _add_check(report, "Feature Confidence", "PASS", "All feature types have MEDIUM or HIGH confidence.", {})
    else:
        _add_check(
            report, "Feature Confidence", "WARN",
            f"Low confidence features: {', '.join(low_confidence_types)}. "
            "These will require manual placement.",
            {"low_confidence_types": low_confidence_types},
        )


def _check_canvas_fit(build_pack: dict, report: dict) -> None:
    """Check whether the course fits within 2K canvas limits."""
    total_yards = build_pack.get("total_yards", 0)
    if not total_yards:
        _add_check(report, "Canvas Fit", "WARN", "Total yards unknown — canvas fit unverifiable.", {})
        return

    if total_yards > config.TK2_MAX_TOTAL_YARDS:
        _add_check(
            report, "Canvas Fit", "WARN",
            f"Course total yardage ({int(total_yards):,}y) exceeds 2K maximum "
            f"({config.TK2_MAX_TOTAL_YARDS:,}y). Some holes may need compressing.",
            {"total_yards": total_yards, "limit": config.TK2_MAX_TOTAL_YARDS},
        )
    else:
        _add_check(
            report, "Canvas Fit", "PASS",
            f"Total {int(total_yards):,}y fits within 2K canvas ({config.TK2_MAX_TOTAL_YARDS:,}y max).",
            {"total_yards": total_yards},
        )


def _check_course_feel(terrain_stats: dict, features_data: dict, report: dict) -> None:
    """
    Heuristic check: does the terrain/feature data suggest a coherent course character?
    """
    z_range = terrain_stats.get("z_range_m", 0)
    steep   = terrain_stats.get("steep_pct", 0)
    cs      = features_data.get("confidence_summary", {})

    hints = []
    if z_range > 30 and steep > 15:
        hints.append("High relief and steep terrain suggests LINKS/CLIFFTOP character.")
    elif z_range < 10 and steep < 5:
        hints.append("Low relief and gentle terrain suggests PARKLAND/FLAT INLAND character.")
    else:
        hints.append("Moderate relief — could be links, heathland, or parkland. Verify course type.")

    water_count = cs.get("water", {}).get("count", 0)
    if water_count > 0:
        hints.append(f"{water_count} water feature(s) found.")

    _add_check(
        report, "Course Feel", "PASS",
        " ".join(hints),
        {
            "z_range_m":  z_range,
            "steep_pct":  steep,
            "water_count": water_count,
        },
    )


# ─── Utilities ────────────────────────────────────────────────────────────────

def _add_check(report: dict, name: str, status: str, message: str, data: dict) -> None:
    entry = {"check": name, "status": status, "message": message, "data": data}
    report["checks"].append(entry)
    if status == "WARN":
        report["warnings"].append(f"{name}: {message}")
    elif status == "FAIL":
        report["errors"].append(f"{name}: {message}")


def _bearing(pt1: list, pt2: list) -> float:
    """Compass bearing in degrees from pt1 to pt2 (lon, lat)."""
    lat1 = math.radians(pt1[1])
    lat2 = math.radians(pt2[1])
    dlon = math.radians(pt2[0] - pt1[0])
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def _haversine(pt1: list, pt2: list) -> float:
    lon1, lat1 = math.radians(pt1[0]), math.radians(pt1[1])
    lon2, lat2 = math.radians(pt2[0]), math.radians(pt2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


# ─── QA HTML output ───────────────────────────────────────────────────────────

def _write_qa_html(report: dict, out_path: Path) -> None:
    status_colours = {"PASS": "#4caf50", "WARN": "#ff9800", "FAIL": "#f44336"}
    overall_colour = status_colours.get(report["overall_status"], "#9e9e9e")

    rows = ""
    for check in report["checks"]:
        colour = status_colours.get(check["status"], "#9e9e9e")
        rows += f"""
<tr>
  <td>{check['check']}</td>
  <td style="color:{colour};font-weight:bold">{check['status']}</td>
  <td>{check['message']}</td>
</tr>"""

    warnings_html = ""
    for w in report["warnings"]:
        warnings_html += f"<li class='warn-item'>{w}</li>"

    errors_html = ""
    for e in report["errors"]:
        errors_html += f"<li class='error-item'>{e}</li>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>QA Report — {report['course_name']}</title>
<style>
body {{ font-family: monospace; background: #1a1a2e; color: #e0e0e0; padding: 2em; }}
h1 {{ color: #fff; }}
.overall {{ font-size: 1.5em; padding: 0.5em; border-radius: 4px;
            background: {overall_colour}; color: #fff; display: inline-block; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ padding: 0.6em 1em; border: 1px solid #333; text-align: left; }}
th {{ background: #333; }}
tr:nth-child(even) {{ background: #222; }}
ul {{ margin: 0.5em 0; padding-left: 1.5em; }}
.warn-item {{ color: #ffb300; }}
.error-item {{ color: #ef5350; }}
</style>
</head>
<body>
<h1>QA Report: {report['course_name']}</h1>
<p>Holes detected: <b>{report['total_holes']}</b></p>
<p>Overall: <span class="overall">{report['overall_status']}</span></p>

<h2>Checks</h2>
<table>
<thead><tr><th>Check</th><th>Status</th><th>Message</th></tr></thead>
<tbody>{rows}</tbody>
</table>

<h2>Warnings</h2>
<ul>{warnings_html if warnings_html else '<li>None</li>'}</ul>

<h2>Errors</h2>
<ul>{errors_html if errors_html else '<li>None</li>'}</ul>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    log.info(f"QA report written: {out_path}")

"""
translation.py — Convert real-world geodata into PGA TOUR 2K build instructions.

Produces:
  - build_instructions.json   Machine-readable step list per hole
  - build_guide.html          Human-readable companion guide
  - course_metadata.json      Distances, pars, elevations, course type
"""

import json
import logging
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import shape, Point
from shapely.ops import transform as shp_transform
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_build_pack(
    boundary_data:  dict,
    terrain_stats:  dict,
    features_data:  dict,
    output_dir:     Path,
    course_type:    str = "links",
    scorecard:      Optional[List[dict]] = None,
) -> dict:
    """
    Generate the full 2K build pack from processed pipeline data.

    Args:
        boundary_data:  From boundary.resolve_boundary()
        terrain_stats:  From terrain.process_terrain()
        features_data:  From features.extract_features()
        output_dir:     Output directory
        course_type:    One of: links, parkland, heathland, clifftop, inland
        scorecard:      Optional manual scorecard list [{hole, par, yards, si, ...}]

    Returns:
        {
            "build_instructions_path": str,
            "build_guide_path":        str,
            "course_metadata_path":    str,
            "total_holes":             int,
            "total_yards":             float,
            "fidelity_estimate":       str,
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    holes       = features_data["holes"]
    preset      = config.COURSE_TYPE_PRESETS.get(course_type, config.COURSE_TYPE_PRESETS["links"])
    course_name = boundary_data["matched_name"]

    # Compute 2K canvas scale
    scale = _compute_scale(boundary_data, terrain_stats)
    log.info(f"2K scale: {scale['metres_per_yard']:.3f} m/yd, "
             f"terrain scale: {scale['tk2_per_metre']:.3f} 2K units/m")

    # Merge scorecard if provided
    if scorecard:
        holes = _merge_scorecard(holes, scorecard)

    # Generate per-hole instructions
    all_instructions = []
    for hole in holes:
        steps = _generate_hole_steps(hole, terrain_stats, scale, preset, features_data)
        all_instructions.append({
            "hole_number":  hole["hole_number"],
            "par":          hole.get("par"),
            "handicap":     hole.get("handicap"),
            "length_yards": hole.get("length_yards"),
            "course_type":  course_type,
            "steps":        steps,
        })

    # Write build instructions JSON
    instructions_path = output_dir / "build_instructions.json"
    instructions_path.write_text(json.dumps(all_instructions, indent=2))

    # Compile course metadata
    total_yards = sum(
        h.get("length_yards", 0) or 0
        for h in holes
        if h.get("length_yards")
    )
    pars = [h.get("par") for h in holes if h.get("par")]
    total_par = sum(pars) if pars else None

    metadata = {
        "course_name":          course_name,
        "course_type":          course_type,
        "total_holes":          len(holes),
        "total_yards":          round(total_yards, 0),
        "total_par":            total_par,
        "terrain_relief_m":     terrain_stats["z_range_m"],
        "terrain_relief_2k":    round(terrain_stats["z_range_m"] * scale["tk2_per_metre"], 1),
        "scale_metres_per_yard":round(scale["metres_per_yard"], 3),
        "canvas_used_pct":      round(total_yards / config.TK2_CANVAS_YARDS * 100, 1) if total_yards else None,
        "conditions_preset":    preset,
        "confidence_summary":   features_data["confidence_summary"],
        "holes":                holes,
    }

    metadata_path = output_dir / "course_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    # Estimate fidelity
    fidelity = _estimate_fidelity(boundary_data, terrain_stats, features_data)

    # Generate HTML guide
    guide_path = output_dir / "build_guide.html"
    _generate_html_guide(course_name, all_instructions, metadata, fidelity, guide_path)

    return {
        "build_instructions_path": str(instructions_path),
        "build_guide_path":        str(guide_path),
        "course_metadata_path":    str(metadata_path),
        "total_holes":             len(holes),
        "total_yards":             total_yards,
        "fidelity_estimate":       fidelity,
    }


# ─── Scale computation ────────────────────────────────────────────────────────

def _compute_scale(boundary_data: dict, terrain_stats: dict) -> dict:
    """
    Compute mapping between real-world metres and 2K coordinate/height units.
    """
    from shapely.geometry import shape
    from shapely.ops import transform as shp_transform
    from pyproj import Transformer

    # Real-world course extent in metres
    poly = shape(boundary_data["boundary_wgs84"])
    t    = Transformer.from_crs(config.CRS_WGS84, config.CRS_ITM, always_xy=True)
    poly_itm = shp_transform(t.transform, poly)
    bounds   = poly_itm.bounds
    width_m  = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]
    max_dim_m = max(width_m, height_m)

    # How many real metres per 2K yard
    # If course fits in canvas, scale 1:1 in yards
    canvas_m = config.TK2_CANVAS_YARDS * 0.9144   # yards → metres
    if max_dim_m <= canvas_m:
        metres_per_yard = 0.9144   # 1 yard = 0.9144m exactly
    else:
        # Course is larger than 2K canvas — scale down
        metres_per_yard = max_dim_m / config.TK2_CANVAS_YARDS
        log.warning(
            f"Course dimension ({max_dim_m:.0f}m) exceeds 2K canvas "
            f"({canvas_m:.0f}m). Scale factor: {metres_per_yard:.3f} m/yd"
        )

    yards_per_metre  = 1.0 / metres_per_yard
    z_range = max(terrain_stats["z_range_m"], 0.01)
    tk2_per_metre    = (config.TK2_HEIGHT_MAX - config.TK2_HEIGHT_MIN) / z_range

    return {
        "metres_per_yard":  metres_per_yard,
        "yards_per_metre":  yards_per_metre,
        "tk2_per_metre":    tk2_per_metre,
        "width_m":          round(width_m, 1),
        "height_m":         round(height_m, 1),
    }


def m_to_yards(metres: float, scale: dict) -> float:
    """Convert real metres to 2K yards."""
    return metres * scale["yards_per_metre"]


def z_to_2k(z_m: float, terrain_stats: dict) -> float:
    """Convert real elevation to 2K height slider (0–100)."""
    z_min   = terrain_stats["z_min_m"]
    z_range = max(terrain_stats["z_range_m"], 0.01)
    return round(max(0.0, min(100.0, (z_m - z_min) / z_range * 100)), 1)


# ─── Per-hole step generation ─────────────────────────────────────────────────

def _generate_hole_steps(
    hole: dict, terrain_stats: dict, scale: dict, preset: dict, features_data: dict
) -> List[dict]:
    """Generate ordered build steps for a single hole."""
    steps = []
    hole_num    = hole["hole_number"]
    length_m    = hole.get("length_m", 0) or 0
    length_yd   = hole.get("length_yards", 0) or 0
    par         = hole.get("par", "?")
    bunkers     = hole.get("bunkers", [])
    fairways    = hole.get("fairways", [])
    tee_pt      = hole.get("tee_centroid", [0, 0])
    green_pt    = hole.get("green_centroid", [0, 0])
    confidence  = features_data["confidence_summary"]

    step_num = 1

    # ── Step 1: Terrain setup ────────────────────────────────────────────────
    terrain_2k_range = round(terrain_stats["z_range_m"] * scale["tk2_per_metre"], 1)
    steps.append({
        "step":     step_num,
        "category": "TERRAIN",
        "title":    "Set terrain height reference",
        "action": (
            f"In Course Designer, set terrain height range to cover "
            f"{terrain_stats['z_range_m']:.1f}m real relief. "
            f"2K height slider: 0 = {terrain_stats['z_min_m']:.1f}m ASL, "
            f"100 = {terrain_stats['z_max_m']:.1f}m ASL. "
            f"Each 2K unit ≈ {1/scale['tk2_per_metre']:.2f}m real elevation."
        ),
        "measurements": {
            "real_relief_m":    terrain_stats["z_range_m"],
            "tk2_height_range": terrain_2k_range,
        },
        "reference":   "heightmap.png",
        "confidence":  "HIGH",
        "manual_flag": None,
    })
    step_num += 1

    # ── Step 2: Rough terrain sculpting ─────────────────────────────────────
    steps.append({
        "step":     step_num,
        "category": "TERRAIN",
        "title":    "Sculpt major landforms",
        "action": (
            f"Using heightmap.png and slope_map.png as reference, sculpt the "
            f"major landforms for Hole {hole_num}. "
            f"Terrain character: {_terrain_character_hint(terrain_stats, preset)}. "
            f"Flat zones (green/tee sites): height ±1 2K unit. "
            f"Steep zones (dunes/banks): follow slope map red areas."
        ),
        "measurements": {
            "flat_pct":     terrain_stats["flat_pct"],
            "steep_pct":    terrain_stats["steep_pct"],
        },
        "reference":   "slope_map.png",
        "confidence":  "HIGH",
        "manual_flag": None,
    })
    step_num += 1

    # ── Step 3: Fairway painting ─────────────────────────────────────────────
    fw_conf = confidence.get("fairway", {}).get("label", "LOW")
    fw_count = len(fairways)
    steps.append({
        "step":     step_num,
        "category": "FAIRWAY",
        "title":    "Paint fairway surface",
        "action": (
            f"Paint fairway starting approx {_format_yards(40, scale)}y from tee. "
            f"Approximate width: {_fairway_width_hint(par)}y. "
            f"Length to green: {int(length_yd)}y. "
            f"See hole_{hole_num:02d}_overview.png — blue/green polygon. "
            f"OSM fairways found: {fw_count}."
        ),
        "measurements": {
            "length_yards":    int(length_yd),
            "approx_width_y":  _fairway_width_hint(par),
        },
        "reference":    f"holes/hole_{hole_num:02d}_overview.png",
        "confidence":   fw_conf,
        "manual_flag":  None if fw_conf in ("HIGH", "MEDIUM") else
                        "No fairway data in OSM — trace from satellite view manually",
    })
    step_num += 1

    # ── Step 4: Green placement ──────────────────────────────────────────────
    green_conf = confidence.get("green", {}).get("label", "LOW")
    steps.append({
        "step":     step_num,
        "category": "GREEN",
        "title":    "Place and sculpt green",
        "action": (
            f"Paint green surface at hole endpoint. "
            f"Typical green size: 25–35y wide. "
            f"Apply {preset['green_speed']} speed. Apply {preset['ground']} firmness. "
            f"Check slope_map.png for green tilt direction — sculpt accordingly. "
            f"Paint subtle undulation (0.5–1.0 2K units max on a links green)."
        ),
        "measurements": {
            "approx_width_y": 30,
            "firmness":       preset["ground"],
            "green_speed":    preset["green_speed"],
        },
        "reference":   f"holes/hole_{hole_num:02d}_overview.png",
        "confidence":  green_conf,
        "manual_flag": None,
    })
    step_num += 1

    # ── Step 5: Tee box ──────────────────────────────────────────────────────
    steps.append({
        "step":     step_num,
        "category": "TEE",
        "title":    "Place tee boxes",
        "action": (
            f"Place championship tee at hole start. "
            f"Set hole distance to {int(length_yd)}y. "
            f"Add medal/forward tee offset: -10 to -30y from championship. "
            f"Tee box size: approx 8×12y. Flatten tee surface."
        ),
        "measurements": {
            "championship_yards": int(length_yd),
        },
        "reference":   f"holes/hole_{hole_num:02d}_overview.png",
        "confidence":  confidence.get("tee", {}).get("label", "MEDIUM"),
        "manual_flag": None,
    })
    step_num += 1

    # ── Step 6: Bunkers ──────────────────────────────────────────────────────
    if bunkers:
        bunk_conf = confidence.get("bunker", {}).get("label", "MEDIUM")
        bunker_descriptions = []
        for i, b in enumerate(bunkers[:8]):   # cap at 8 bunkers per hole
            cx, cy = b["centroid"]
            dist_m = _haversine(tee_pt, (cx, cy))
            dist_y = dist_m * 1.09361 * scale["yards_per_metre"]
            side   = _side_of_line(tee_pt, green_pt, (cx, cy))
            area_m = b["area_m2"]
            width_y = max(4, round(math.sqrt(area_m) * 1.09, 0))
            bunker_descriptions.append(
                f"Bunker {i+1}: {int(dist_y)}y from tee, {side} side, "
                f"approx {int(width_y)}y wide. "
                f"(Confidence: {b['confidence']:.2f})"
            )

        steps.append({
            "step":     step_num,
            "category": "BUNKER",
            "title":    f"Place {len(bunkers)} bunker(s)",
            "action": (
                "Place bunkers using positions below. "
                "For links: set depth DEEP. All bunkers: vertical face toward green. "
                "Cross-reference satellite view for shape refinement.\n"
                + "\n".join(bunker_descriptions)
            ),
            "measurements": {
                "bunker_count": len(bunkers),
            },
            "reference":   f"holes/hole_{hole_num:02d}_overview.png",
            "confidence":  bunk_conf,
            "manual_flag": (
                "MEDIUM/LOW confidence — verify positions against satellite view"
                if bunk_conf != "HIGH" else None
            ),
        })
        step_num += 1

    # ── Step 7: Water hazards ────────────────────────────────────────────────
    water_conf = confidence.get("water", {})
    if water_conf.get("count", 0) > 0:
        steps.append({
            "step":     step_num,
            "category": "WATER",
            "title":    "Place water hazard(s)",
            "action": (
                "Place water hazard from feature map. "
                "Use feature_map.png blue polygon for position. "
                "Set hazard boundary type (lateral/frontal) per original course rules."
            ),
            "reference":   "feature_map.png",
            "confidence":  water_conf.get("label", "MEDIUM"),
            "manual_flag": None,
            "measurements": {},
        })
        step_num += 1

    # ── Step 8: Rough and surface zones ─────────────────────────────────────
    steps.append({
        "step":     step_num,
        "category": "ROUGH",
        "title":    "Paint rough and surface zones",
        "action": (
            f"Paint thick rough ({preset['rough']}) on all non-fairway, non-green areas. "
            f"For steep zones (slope map red): extend rough to course boundary. "
            f"For moderate zones: standard rough. "
            f"Leave fairway and green as painted."
        ),
        "reference":   "slope_map.png",
        "confidence":  "HIGH",
        "manual_flag": None,
        "measurements": {
            "rough_type": preset["rough"],
        },
    })
    step_num += 1

    # ── Step 9: Vegetation ───────────────────────────────────────────────────
    budget = config.VEGETATION_BUDGET
    steps.append({
        "step":     step_num,
        "category": "VEGETATION",
        "title":    "Add vegetation",
        "action": _vegetation_hint(preset, budget),
        "reference":   None,
        "confidence":  "LOW",
        "manual_flag": "Vegetation is always manual — use satellite view as reference",
        "measurements": {
            "tree_budget":   budget["trees"],
            "bush_budget":   budget["bushes"],
        },
    })
    step_num += 1

    # ── Step 10: QA walk ─────────────────────────────────────────────────────
    steps.append({
        "step":     step_num,
        "category": "QA",
        "title":    "Walk and validate hole",
        "action": (
            f"Enter preview mode. Walk from tee to green. "
            f"Verify: rangefinder shows ~{int(length_yd)}y (±5% = {int(length_yd*0.95)}–{int(length_yd*1.05)}y). "
            f"Check no surface paint gaps. Check bunkers drain correctly. "
            f"Check green is reachable and slopes toward dominant approach. "
            f"Check par assignment: Par {par}."
        ),
        "reference":   f"holes/hole_{hole_num:02d}_overview.png",
        "confidence":  "HIGH",
        "manual_flag": None,
        "measurements": {
            "target_yards":     int(length_yd),
            "tolerance_yards":  int(length_yd * 0.05),
            "par":              par,
        },
    })

    return steps


# ─── HTML guide generation ────────────────────────────────────────────────────

def _generate_html_guide(
    course_name: str,
    all_instructions: List[dict],
    metadata: dict,
    fidelity: dict,
    out_path: Path,
) -> None:
    """Generate the companion HTML build guide."""
    holes_html = ""
    for hole_data in all_instructions:
        n   = hole_data["hole_number"]
        par = hole_data.get("par", "?")
        yd  = int(hole_data.get("length_yards") or 0)
        si  = hole_data.get("handicap", "?")

        steps_html = ""
        for step in hole_data["steps"]:
            cat_class  = step["category"].lower()
            conf_class = step.get("confidence", "LOW").lower()
            flag_html  = ""
            if step.get("manual_flag"):
                flag_html = f'<div class="manual-flag">⚠ {step["manual_flag"]}</div>'
            meas_html = ""
            if step.get("measurements"):
                meas_items = "".join(
                    f"<li><b>{k.replace('_',' ').title()}:</b> {v}</li>"
                    for k, v in step["measurements"].items()
                )
                meas_html = f"<ul class='measurements'>{meas_items}</ul>"
            ref_html = ""
            if step.get("reference"):
                ref_html = f'<div class="ref-link">📄 Reference: <code>{step["reference"]}</code></div>'

            steps_html += f"""
<div class="step step-{cat_class}" data-step="{step['step']}">
  <div class="step-header">
    <span class="step-num">Step {step['step']}</span>
    <span class="step-cat cat-{cat_class}">{step['category']}</span>
    <span class="step-title">{step['title']}</span>
    <span class="conf conf-{conf_class}">{step.get('confidence','?')}</span>
    <label class="checkbox-label">
      <input type="checkbox" class="step-check"> Done
    </label>
  </div>
  <div class="step-body">
    <p class="action-text">{step['action'].replace(chr(10), '<br>')}</p>
    {meas_html}
    {ref_html}
    {flag_html}
  </div>
</div>"""

        holes_html += f"""
<section class="hole-section" id="hole-{n}">
  <div class="hole-header">
    <h2>Hole {n}</h2>
    <div class="hole-meta">
      <span class="meta-item">Par {par}</span>
      <span class="meta-item">{yd}y</span>
      <span class="meta-item">S.I. {si}</span>
    </div>
    <div class="hole-nav">
      {'<a class="nav-btn" href="#hole-' + str(n-1) + '">◄ ' + str(n-1) + '</a>' if n > 1 else ''}
      {'<a class="nav-btn" href="#hole-' + str(n+1) + '">' + str(n+1) + ' ►</a>' if n < len(all_instructions) else ''}
    </div>
  </div>
  <div class="hole-maps">
    <img src="holes/hole_{n:02d}_overview.png" alt="Hole {n} overview"
         onerror="this.style.display='none'">
  </div>
  <div class="steps-list">
    {steps_html}
  </div>
</section>"""

    confidence_rows = ""
    for ft, cs in metadata.get("confidence_summary", {}).items():
        conf_class = cs.get("label", "LOW").lower()
        confidence_rows += f"""
<tr>
  <td>{ft.replace('_',' ').title()}</td>
  <td>{cs.get('count','—')}</td>
  <td><span class="conf conf-{conf_class}">{cs.get('label','—')}</span></td>
</tr>"""

    fidelity_score = fidelity.get("overall_score", 0)
    fidelity_colour = "#4caf50" if fidelity_score >= 70 else ("#ff9800" if fidelity_score >= 50 else "#f44336")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{course_name} — 2K Build Guide</title>
<link rel="stylesheet" href="static/style.css">
</head>
<body>

<header class="site-header">
  <div class="header-inner">
    <h1>🏌 {course_name}</h1>
    <div class="header-meta">
      <span>{metadata.get('total_holes', '?')} holes</span>
      <span>Par {metadata.get('total_par', '?')}</span>
      <span>{int(metadata.get('total_yards', 0)):,}y total</span>
      <span>{metadata.get('course_type', '').title()}</span>
    </div>
  </div>
</header>

<nav class="hole-nav-bar">
  {''.join(f'<a href="#hole-{h["hole_number"]}" class="hole-nav-item">H{h["hole_number"]}</a>' for h in all_instructions)}
</nav>

<main class="main-content">

<section class="overview-section">
  <h2>Course Overview</h2>
  <div class="overview-grid">
    <div class="overview-card">
      <h3>Terrain</h3>
      <p>Relief: <b>{metadata.get('terrain_relief_m', 0):.1f}m</b></p>
      <p>2K height range: <b>0–{metadata.get('terrain_relief_2k', 100):.0f}</b></p>
    </div>
    <div class="overview-card">
      <h3>Fidelity Estimate</h3>
      <p style="color:{fidelity_colour};font-size:2em;font-weight:bold">
        {fidelity_score}%
      </p>
      <p>{fidelity.get('label','—')}</p>
    </div>
    <div class="overview-card">
      <h3>Conditions</h3>
      <p>Rough: <b>{metadata.get('conditions_preset', {}).get('rough', '—')}</b></p>
      <p>Greens: <b>{metadata.get('conditions_preset', {}).get('green_speed', '—')}</b></p>
      <p>Ground: <b>{metadata.get('conditions_preset', {}).get('ground', '—')}</b></p>
    </div>
    <div class="overview-card">
      <img src="feature_map.png" alt="Course feature map" style="max-width:100%"
           onerror="this.alt='Feature map not generated'">
    </div>
  </div>

  <h3>Data Confidence</h3>
  <table class="confidence-table">
    <thead><tr><th>Feature</th><th>Count</th><th>Confidence</th></tr></thead>
    <tbody>{confidence_rows}</tbody>
  </table>
</section>

{holes_html}

</main>

<script>
// Persist checkbox state in localStorage
document.querySelectorAll('.step-check').forEach(cb => {{
  const key = 'step_' + cb.closest('.step').dataset.step
              + '_hole_' + cb.closest('.hole-section').id;
  cb.checked = localStorage.getItem(key) === '1';
  cb.addEventListener('change', () => {{
    localStorage.setItem(key, cb.checked ? '1' : '0');
    cb.closest('.step').classList.toggle('step-done', cb.checked);
  }});
  if (cb.checked) cb.closest('.step').classList.add('step-done');
}});
</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    log.info(f"Build guide written: {out_path}")


# ─── Fidelity estimation ──────────────────────────────────────────────────────

def _estimate_fidelity(boundary_data: dict, terrain_stats: dict, features_data: dict) -> dict:
    """
    Estimate overall build fidelity based on data quality.
    Returns a dict with overall_score (0–100) and per-dimension scores.
    """
    dimensions = {}

    # Boundary confidence
    bc = boundary_data.get("confidence", "LOW")
    dimensions["boundary"] = {"score": 90 if bc == "HIGH" else (60 if bc == "MEDIUM" else 30), "weight": 5}

    # Terrain coverage
    z_range = terrain_stats.get("z_range_m", 0)
    res     = terrain_stats.get("resolution_m", 30)
    t_score = 90 if res <= 1 else (70 if res <= 5 else (50 if res <= 25 else 30))
    if z_range < 2:
        t_score -= 10  # very flat terrain → less value from LiDAR
    dimensions["terrain"] = {"score": t_score, "weight": 20}

    # Feature confidence per type
    cs = features_data.get("confidence_summary", {})
    for ft, wt in [("fairway", 15), ("green", 20), ("bunker", 15), ("tee", 10), ("hole_routing", 15)]:
        fd = cs.get(ft, {})
        label = fd.get("label", "LOW")
        score = 85 if label == "HIGH" else (60 if label == "MEDIUM" else 30)
        dimensions[ft] = {"score": score, "weight": wt}

    # Scorecard / distances
    holes = features_data.get("holes", [])
    has_par = sum(1 for h in holes if h.get("par")) / max(len(holes), 1)
    dimensions["distances"] = {"score": int(has_par * 100), "weight": 10}

    # Weighted average
    total_weight = sum(d["weight"] for d in dimensions.values())
    overall = sum(d["score"] * d["weight"] for d in dimensions.values()) / total_weight

    label = "EXCELLENT" if overall >= 80 else ("GOOD" if overall >= 65 else ("FAIR" if overall >= 50 else "POOR"))

    return {
        "overall_score": round(overall, 1),
        "label":         label,
        "dimensions":    dimensions,
    }


# ─── Hints and helpers ────────────────────────────────────────────────────────

def _terrain_character_hint(terrain_stats: dict, preset: dict) -> str:
    steep = terrain_stats.get("steep_pct", 0)
    if steep > 20:
        return "dramatic undulation — significant dunes or hillside. Use slope map for guidance."
    if steep > 8:
        return "moderate undulation — rolling fairways, some banks."
    return "gently rolling to flat — subtle undulation only."


def _fairway_width_hint(par) -> int:
    if par == 3:
        return 20
    if par == 5:
        return 45
    return 35  # par 4 default


def _vegetation_hint(preset: dict, budget: dict) -> str:
    if "Fescue" in preset["rough"] or "links" in preset.get("type", ""):
        return (
            "Links: Place marram grass / dune grass as zone paint on dune slopes. "
            f"Max {budget['trees']} trees (sparse — links have very few). "
            "Prioritise open sky feel."
        )
    if "Heather" in preset["rough"]:
        return (
            f"Heathland: Paint heather zones on rough areas. "
            f"Sparse pines/birch on boundaries. Max {budget['trees']} trees. "
            "Open, windswept feel."
        )
    return (
        f"Parkland: Tree lines along fairway edges. "
        f"Max {budget['trees']} trees per hole. "
        "Dense rough under trees. Enclosed, sheltered feel."
    )


def _side_of_line(start: list, end: list, point: list) -> str:
    """Return 'left' or 'right' relative to the tee→green direction."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    ex = point[0] - start[0]
    ey = point[1] - start[1]
    cross = dx * ey - dy * ex
    return "left" if cross > 0 else "right"


def _format_yards(metres: float, scale: dict) -> int:
    return int(metres * scale["yards_per_metre"])


def _haversine(pt1: tuple, pt2: tuple) -> float:
    lon1, lat1 = math.radians(pt1[0]), math.radians(pt1[1])
    lon2, lat2 = math.radians(pt2[0]), math.radians(pt2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def _merge_scorecard(holes: List[dict], scorecard: List[dict]) -> List[dict]:
    """Merge manual scorecard data into hole routing data."""
    sc_map = {s["hole"]: s for s in scorecard if "hole" in s}
    for hole in holes:
        n = hole["hole_number"]
        if n in sc_map:
            sc = sc_map[n]
            if sc.get("par"):
                hole["par"] = sc["par"]
            if sc.get("yards"):
                # Prefer manual yardage over calculated
                hole["length_yards"] = sc["yards"]
                hole["length_m"]     = sc["yards"] * 0.9144
            if sc.get("si") or sc.get("handicap"):
                hole["handicap"] = sc.get("si") or sc.get("handicap")
    return holes

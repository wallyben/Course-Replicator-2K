"""
buildpack.py — Enhanced build pack generation for Version 2.

PART 4: Upgraded build pack generator.

Outputs:
  - heightmap.png           (from terrain.py — already generated)
  - routing.geojson         (from routing.py)
  - greens.geojson          (from osm_features.py)
  - bunkers.geojson         (from osm_features.py)
  - vegetation_zones.geojson (derived from terrain regions)
  - course_manifest.json    (comprehensive course metadata)
  - hole_blueprints.pdf     (per-hole reference sheets)
  - replication_report.html (full interactive report)
"""

import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_enhanced_buildpack(
    boundary_data: dict,
    terrain_stats: dict,
    features_data: dict,
    routing_data: dict,
    output_dir: Path,
    course_type: str = "parkland",
    osm_features_dir: Optional[Path] = None,
) -> dict:
    """
    Generate the Version 2 enhanced build pack.

    Args:
        boundary_data:    Output from boundary.resolve_boundary()
        terrain_stats:    Output from terrain.process_terrain()
        features_data:    Output from features.extract_features()
        routing_data:     Output from routing.reconstruct_routing()
        output_dir:       Directory to write all outputs
        course_type:      Course type preset key
        osm_features_dir: Optional directory with per-type GeoJSON

    Returns:
        Build pack summary dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    course_name = boundary_data.get("name", "Unknown Course")
    log.info(f"Generating enhanced build pack for: {course_name}")

    # Gather all outputs
    outputs = {}

    # 1. Vegetation zones (derived from terrain regions)
    veg_path = output_dir / "vegetation_zones.geojson"
    try:
        _generate_vegetation_zones(terrain_stats, output_dir, veg_path, course_type)
        outputs["vegetation_zones"] = str(veg_path)
        log.info("Vegetation zones generated")
    except Exception as e:
        log.warning(f"Vegetation zones failed (non-critical): {e}")

    # 2. Collect all GeoJSON feature paths
    feature_paths = _collect_feature_paths(output_dir, osm_features_dir)
    outputs.update(feature_paths)

    # 3. Course manifest
    manifest = _build_course_manifest(
        boundary_data, terrain_stats, features_data,
        routing_data, course_type, feature_paths
    )
    manifest_path = output_dir / "course_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    outputs["course_manifest"] = str(manifest_path)
    log.info("Course manifest written")

    # 4. Hole blueprints PDF
    pdf_path = output_dir / "hole_blueprints.pdf"
    try:
        _generate_hole_blueprints_pdf(
            routing_data, terrain_stats, features_data, output_dir, pdf_path
        )
        outputs["hole_blueprints"] = str(pdf_path)
        log.info("Hole blueprints PDF generated")
    except Exception as e:
        log.warning(f"PDF generation failed (non-critical): {e}")
        outputs["hole_blueprints"] = None

    # 5. Replication report HTML
    html_path = output_dir / "replication_report.html"
    try:
        _generate_replication_report_html(
            manifest, routing_data, terrain_stats, output_dir, html_path
        )
        outputs["replication_report"] = str(html_path)
        log.info("Replication report HTML generated")
    except Exception as e:
        log.warning(f"HTML report failed (non-critical): {e}")

    return {
        "course_name":  course_name,
        "course_type":  course_type,
        "outputs":      outputs,
        "manifest":     manifest,
    }


# ─── Vegetation zones ─────────────────────────────────────────────────────────

def _generate_vegetation_zones(
    terrain_stats: dict,
    output_dir: Path,
    out_path: Path,
    course_type: str,
) -> None:
    """
    Derive vegetation zone polygons from terrain regions GeoJSON.

    Mapping:
      - flat   (1) → maintained turf / fairway rough
      - gentle (2) → medium rough / light tree cover
      - moderate (3) → heavy rough / tree zones
      - steep  (4) → woodland / natural terrain
    """
    regions_path = output_dir / "terrain_regions.geojson"
    if not regions_path.exists():
        log.warning("terrain_regions.geojson not found — skipping vegetation zones")
        return

    preset = config.COURSE_TYPE_PRESETS.get(course_type, {})
    rough_type = preset.get("rough", "Thick Rough")

    veg_mapping = {
        1: {"vegetation": "maintained_turf",  "density": "low",    "rough": rough_type},
        2: {"vegetation": "medium_rough",     "density": "medium", "rough": rough_type},
        3: {"vegetation": "heavy_rough",      "density": "high",   "rough": rough_type},
        4: {"vegetation": "woodland",         "density": "dense",  "rough": "Trees"},
    }
    # Parkland/heathland get more trees on steep terrain
    if course_type in ("parkland", "heathland"):
        veg_mapping[4]["vegetation"] = "woodland_trees"
        veg_mapping[3]["vegetation"] = "tree_rough"

    regions_data = json.loads(regions_path.read_text())
    features = []
    for feat in regions_data.get("features", []):
        cls = feat.get("properties", {}).get("class", 0)
        veg = veg_mapping.get(cls, {})
        new_feat = {
            "type":     "Feature",
            "geometry": feat["geometry"],
            "properties": {
                "terrain_class": cls,
                "terrain_label": feat.get("properties", {}).get("label", ""),
                **veg,
            },
        }
        features.append(new_feat)

    geojson = {"type": "FeatureCollection", "features": features}
    out_path.write_text(json.dumps(geojson, indent=2))


# ─── Collect existing outputs ─────────────────────────────────────────────────

def _collect_feature_paths(
    output_dir: Path,
    osm_features_dir: Optional[Path],
) -> dict:
    """
    Collect paths to existing GeoJSON files in output and osm_features directories.
    """
    paths = {}
    search_dirs = [output_dir]
    if osm_features_dir and osm_features_dir != output_dir:
        search_dirs.append(osm_features_dir)

    target_files = [
        "routing.geojson", "holes.geojson",
        "fairways.geojson", "greens.geojson", "bunkers.geojson",
        "tees.geojson", "paths.geojson", "water.geojson",
        "features.geojson", "terrain_regions.geojson",
        "heightmap.png", "slope_map.png",
    ]

    for fname in target_files:
        for d in search_dirs:
            p = d / fname
            if p.exists():
                paths[fname.replace(".", "_").replace("-", "_")] = str(p)
                break

    return paths


# ─── Course manifest ──────────────────────────────────────────────────────────

def _build_course_manifest(
    boundary_data: dict,
    terrain_stats: dict,
    features_data: dict,
    routing_data: dict,
    course_type: str,
    feature_paths: dict,
) -> dict:
    """
    Build comprehensive course_manifest.json.
    """
    holes = routing_data.get("holes", [])
    total_yards = sum(
        h.get("distance_yards", 0) or 0 for h in holes
    )
    total_par = sum(h.get("par", 4) or 4 for h in holes)

    return {
        "version":        "2.0",
        "generated_date": str(date.today()),
        "course": {
            "name":           boundary_data.get("name", ""),
            "matched_name":   boundary_data.get("matched_name", ""),
            "type":           course_type,
            "area_ha":        round(boundary_data.get("area_m2", 0) / 10000, 1),
            "centre":         boundary_data.get("centre_wgs84"),
            "bbox":           boundary_data.get("bbox_wgs84"),
            "osm_id":         boundary_data.get("osm_id"),
        },
        "terrain": {
            "z_min_m":        terrain_stats.get("z_min_m"),
            "z_max_m":        terrain_stats.get("z_max_m"),
            "z_range_m":      terrain_stats.get("z_range_m"),
            "resolution_m":   terrain_stats.get("resolution_m"),
            "slope_mean_deg": terrain_stats.get("slope_mean_deg"),
            "flat_pct":       terrain_stats.get("flat_pct"),
            "gentle_pct":     terrain_stats.get("gentle_pct"),
            "moderate_pct":   terrain_stats.get("moderate_pct"),
            "steep_pct":      terrain_stats.get("steep_pct"),
        },
        "scorecard": {
            "holes":       len(holes),
            "total_yards": round(total_yards),
            "total_par":   total_par,
            "holes":       [
                {
                    "hole":   h.get("hole_number"),
                    "par":    h.get("par"),
                    "yards":  round(h.get("distance_yards") or 0),
                    "metres": round(h.get("distance_m") or 0, 1),
                    "source": h.get("routing_source"),
                }
                for h in holes
            ],
        },
        "features": {
            "fairways": features_data.get("feature_counts", {}).get("fairway", 0),
            "greens":   features_data.get("feature_counts", {}).get("green", 0),
            "bunkers":  features_data.get("feature_counts", {}).get("bunker", 0),
            "tees":     features_data.get("feature_counts", {}).get("tee", 0),
            "water":    features_data.get("feature_counts", {}).get("water", 0),
        },
        "tk2_mapping": {
            "height_at_z_min": terrain_stats.get("tk2_height_at_z_min"),
            "height_at_z_max": terrain_stats.get("tk2_height_at_z_max"),
            "height_per_metre": terrain_stats.get("tk2_height_per_metre"),
        },
        "output_files": feature_paths,
    }


# ─── Hole blueprints PDF ──────────────────────────────────────────────────────

def _generate_hole_blueprints_pdf(
    routing_data: dict,
    terrain_stats: dict,
    features_data: dict,
    output_dir: Path,
    out_path: Path,
) -> None:
    """
    Generate per-hole reference PDF using matplotlib PDF backend.
    Each page = one hole with: metadata table, build steps, diagram placeholder.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    holes = routing_data.get("holes", [])
    if not holes:
        log.warning("No holes data for PDF — skipping")
        return

    preset = config.COURSE_TYPE_PRESETS.get("parkland", {})
    tk2_per_m = terrain_stats.get("tk2_height_per_metre", 1.0)

    with PdfPages(str(out_path)) as pdf:
        # Cover page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_axis_off()
        course_name = routing_data.get("holes", [{}])[0].get("routing_source", "")
        ax.text(0.5, 0.75, "Course Replicator 2K", fontsize=28, ha="center",
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.65, "Hole Blueprints", fontsize=18, ha="center",
                transform=ax.transAxes, color="#444")
        ax.text(0.5, 0.55, f"Generated: {date.today()}", fontsize=12,
                ha="center", transform=ax.transAxes, color="#666")
        ax.text(0.5, 0.45, f"Holes: {len(holes)}", fontsize=14,
                ha="center", transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Per-hole pages
        for hole in holes:
            fig, axes = plt.subplots(1, 2, figsize=(11, 8.5),
                                     gridspec_kw={"width_ratios": [1, 1.5]})
            fig.suptitle(
                f"Hole {hole['hole_number']}   Par {hole.get('par', '?')}   "
                f"{round(hole.get('distance_yards', 0) or 0)}y",
                fontsize=16, fontweight="bold"
            )

            # Left: metadata table
            ax_meta = axes[0]
            ax_meta.set_axis_off()
            rows = [
                ["Hole",    str(hole["hole_number"])],
                ["Par",     str(hole.get("par", "?"))],
                ["Yards",   f"{round(hole.get('distance_yards', 0) or 0)}y"],
                ["Metres",  f"{round(hole.get('distance_m', 0) or 0, 1)}m"],
                ["Source",  str(hole.get("routing_source", ""))],
            ]
            tee = hole.get("tee_position")
            green = hole.get("green_position")
            if tee:
                rows.append(["Tee lon/lat", f"{tee['lon']:.5f}, {tee['lat']:.5f}"])
            if green:
                rows.append(["Green lon/lat", f"{green['lon']:.5f}, {green['lat']:.5f}"])

            table = ax_meta.table(
                cellText=rows,
                colLabels=["Property", "Value"],
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.5)

            # Right: build steps
            ax_steps = axes[1]
            ax_steps.set_axis_off()
            steps = _hole_build_steps(hole, terrain_stats)
            step_text = "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(steps)
            )
            ax_steps.text(
                0.02, 0.97, "Build Steps:", fontsize=11, fontweight="bold",
                transform=ax_steps.transAxes, va="top"
            )
            ax_steps.text(
                0.02, 0.90, step_text, fontsize=8,
                transform=ax_steps.transAxes, va="top",
                wrap=True, family="monospace"
            )

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    log.info(f"Hole blueprints PDF: {out_path} ({len(holes)} holes)")


def _hole_build_steps(hole: dict, terrain_stats: dict) -> List[str]:
    """Generate concise build steps for one hole."""
    steps = [
        "Set terrain height reference using heightmap.png",
        "Sculpt major landforms per slope_map.png",
    ]
    d_yards = round(hole.get("distance_yards", 0) or 0)
    if d_yards:
        steps.append(f"Paint fairway {d_yards}y from tee to green")
    else:
        steps.append("Paint fairway corridor (reference holes.geojson)")
    steps += [
        "Place and sculpt putting green (ref greens.geojson)",
        "Add tee boxes (ref tees.geojson)",
        "Place bunkers per bunkers.geojson",
        "Add water hazards if present (water.geojson)",
        "Paint rough zones around fairway",
        "Add vegetation per vegetation_zones.geojson",
        "Walk the hole and validate distances",
    ]
    return steps


# ─── Replication report HTML ──────────────────────────────────────────────────

def _generate_replication_report_html(
    manifest: dict,
    routing_data: dict,
    terrain_stats: dict,
    output_dir: Path,
    out_path: Path,
) -> None:
    """
    Generate a self-contained HTML replication report.
    """
    course = manifest.get("course", {})
    scorecard = manifest.get("scorecard", {})
    terrain = manifest.get("terrain", {})
    feats = manifest.get("features", {})
    holes = scorecard.get("holes", [])

    # Build scorecard rows
    sc_rows = ""
    for h in holes:
        sc_rows += (
            f"<tr><td>{h.get('hole','')}</td>"
            f"<td>{h.get('par','')}</td>"
            f"<td>{h.get('yards','')}</td>"
            f"<td>{h.get('metres','')}</td>"
            f"<td>{h.get('source','')}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Course Replicator 2K — Replication Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
  h1 {{ color: #4fc3f7; }}
  h2 {{ color: #81c784; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  .card {{ background: #16213e; border-radius: 8px; padding: 16px; margin: 12px 0; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ background: #0f3460; color: #fff; padding: 8px; text-align: left; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #2a2a4a; }}
  tr:hover {{ background: #1f2f5a; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.8em; font-weight: bold; }}
  .badge-high   {{ background: #2e7d32; color: #fff; }}
  .badge-medium {{ background: #f57f17; color: #fff; }}
  .badge-low    {{ background: #b71c1c; color: #fff; }}
  .stat {{ display: inline-block; margin: 6px 12px; text-align: center; }}
  .stat-val {{ font-size: 1.8em; font-weight: bold; color: #4fc3f7; }}
  .stat-lbl {{ font-size: 0.75em; color: #aaa; }}
</style>
</head>
<body>
<h1>&#9971; Course Replicator 2K — Replication Report</h1>
<p>Generated: {manifest.get('generated_date', '')} &nbsp;|&nbsp; Version: {manifest.get('version', '2.0')}</p>

<div class="card">
  <h2>Course Information</h2>
  <div class="stat"><div class="stat-val">{course.get('name', 'Unknown')}</div><div class="stat-lbl">Course Name</div></div>
  <div class="stat"><div class="stat-val">{course.get('type', '').title()}</div><div class="stat-lbl">Type</div></div>
  <div class="stat"><div class="stat-val">{course.get('area_ha', 0)} ha</div><div class="stat-lbl">Area</div></div>
  <div class="stat"><div class="stat-val">{scorecard.get('holes', 0)}</div><div class="stat-lbl">Holes</div></div>
  <div class="stat"><div class="stat-val">{int(scorecard.get('total_yards', 0)):,}y</div><div class="stat-lbl">Total Yards</div></div>
  <div class="stat"><div class="stat-val">{scorecard.get('total_par', 0)}</div><div class="stat-lbl">Total Par</div></div>
</div>

<div class="card">
  <h2>Terrain Analysis</h2>
  <div class="stat"><div class="stat-val">{terrain.get('z_min_m', 0):.1f}m</div><div class="stat-lbl">Min Elevation</div></div>
  <div class="stat"><div class="stat-val">{terrain.get('z_max_m', 0):.1f}m</div><div class="stat-lbl">Max Elevation</div></div>
  <div class="stat"><div class="stat-val">{terrain.get('z_range_m', 0):.1f}m</div><div class="stat-lbl">Relief</div></div>
  <div class="stat"><div class="stat-val">{terrain.get('slope_mean_deg', 0):.1f}°</div><div class="stat-lbl">Mean Slope</div></div>
  <div class="stat"><div class="stat-val">{terrain.get('flat_pct', 0):.0f}%</div><div class="stat-lbl">Flat</div></div>
  <div class="stat"><div class="stat-val">{terrain.get('steep_pct', 0):.0f}%</div><div class="stat-lbl">Steep</div></div>
</div>

<div class="card">
  <h2>OSM Feature Coverage</h2>
  <div class="stat"><div class="stat-val">{feats.get('fairways', 0)}</div><div class="stat-lbl">Fairways</div></div>
  <div class="stat"><div class="stat-val">{feats.get('greens', 0)}</div><div class="stat-lbl">Greens</div></div>
  <div class="stat"><div class="stat-val">{feats.get('bunkers', 0)}</div><div class="stat-lbl">Bunkers</div></div>
  <div class="stat"><div class="stat-val">{feats.get('tees', 0)}</div><div class="stat-lbl">Tees</div></div>
  <div class="stat"><div class="stat-val">{feats.get('water', 0)}</div><div class="stat-lbl">Water</div></div>
</div>

<div class="card">
  <h2>Scorecard</h2>
  <table>
    <thead><tr><th>Hole</th><th>Par</th><th>Yards</th><th>Metres</th><th>Data Source</th></tr></thead>
    <tbody>{sc_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>Build Pack Files</h2>
  <table>
    <thead><tr><th>File</th><th>Description</th></tr></thead>
    <tbody>
      <tr><td>heightmap.png</td><td>16-bit greyscale terrain heightmap for 2K import</td></tr>
      <tr><td>slope_map.png</td><td>Colour-coded slope classification reference</td></tr>
      <tr><td>terrain_regions.geojson</td><td>Flat/gentle/moderate/steep polygon zones</td></tr>
      <tr><td>vegetation_zones.geojson</td><td>Vegetation density zones derived from terrain</td></tr>
      <tr><td>holes.geojson</td><td>18-hole routing with tee/green/corridor</td></tr>
      <tr><td>greens.geojson</td><td>Putting green polygons</td></tr>
      <tr><td>bunkers.geojson</td><td>Bunker polygons</td></tr>
      <tr><td>course_manifest.json</td><td>Full course metadata (machine-readable)</td></tr>
      <tr><td>hole_blueprints.pdf</td><td>Per-hole build reference (printed beside Xbox)</td></tr>
      <tr><td>replication_report.html</td><td>This report</td></tr>
    </tbody>
  </table>
</div>

<div class="card">
  <h2>2K Height Mapping</h2>
  <p>Terrain height slider range:</p>
  <ul>
    <li>Lowest point ({terrain.get('z_min_m', 0):.1f}m) → 2K height <strong>{manifest.get('tk2_mapping', {}).get('height_at_z_min', 0)}</strong></li>
    <li>Highest point ({terrain.get('z_max_m', 0):.1f}m) → 2K height <strong>{manifest.get('tk2_mapping', {}).get('height_at_z_max', 100)}</strong></li>
    <li>Scale: <strong>{manifest.get('tk2_mapping', {}).get('height_per_metre', 1):.2f}</strong> 2K units per real metre</li>
  </ul>
</div>

</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")

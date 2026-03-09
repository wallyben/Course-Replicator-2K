"""
companion/app.py — Local Flask companion web app for beside-Xbox use.

Run: python companion/app.py --course output/old-conna-golf-club
Then open http://localhost:5000 on a tablet or second screen beside the Xbox.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from flask import Flask, render_template, jsonify, send_from_directory, abort

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

log = logging.getLogger(__name__)
app = Flask(__name__)

# ─── Global state ─────────────────────────────────────────────────────────────
COURSE_DIR: Path = None


def _load_json(filename: str) -> dict:
    path = COURSE_DIR / filename
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_instructions() -> list:
    return _load_json("build_instructions.json") or []


def _load_metadata() -> dict:
    return _load_json("course_metadata.json")


def _load_qa() -> dict:
    return _load_json("qa_report.json")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    metadata     = _load_metadata()
    instructions = _load_instructions()
    qa           = _load_qa()

    # Build hole summary list
    holes = []
    for hi in instructions:
        n      = hi["hole_number"]
        par    = hi.get("par", "?")
        yd     = int(hi.get("length_yards") or 0)
        steps  = hi.get("steps", [])
        n_steps = len(steps)
        overview_img = f"holes/hole_{n:02d}_overview.png"
        holes.append({
            "number":    n,
            "par":       par,
            "yards":     yd,
            "n_steps":   n_steps,
            "has_image": (COURSE_DIR / overview_img).exists(),
            "image":     overview_img,
        })

    course_name = metadata.get("course_name", "Unknown Course")
    total_par   = metadata.get("total_par", "?")
    total_yards = int(metadata.get("total_yards", 0))
    course_type = metadata.get("course_type", "links")
    qa_status   = qa.get("overall_status", "—")
    fidelity    = metadata.get("fidelity_estimate", {})
    has_feature_map = (COURSE_DIR / "feature_map.png").exists()
    has_heightmap   = (COURSE_DIR / "heightmap.png").exists()
    has_slope_map   = (COURSE_DIR / "slope_map.png").exists()

    return render_template(
        "index.html",
        course_name=course_name,
        total_par=total_par,
        total_yards=total_yards,
        course_type=course_type,
        holes=holes,
        qa_status=qa_status,
        fidelity=fidelity,
        has_feature_map=has_feature_map,
        has_heightmap=has_heightmap,
        has_slope_map=has_slope_map,
    )


@app.route("/hole/<int:hole_num>")
def hole_view(hole_num: int):
    instructions = _load_instructions()
    metadata     = _load_metadata()

    hole_data = next((h for h in instructions if h["hole_number"] == hole_num), None)
    if hole_data is None:
        abort(404)

    all_holes = sorted([h["hole_number"] for h in instructions])
    idx       = all_holes.index(hole_num)
    prev_hole = all_holes[idx - 1] if idx > 0 else None
    next_hole = all_holes[idx + 1] if idx < len(all_holes) - 1 else None

    course_name = metadata.get("course_name", "Unknown Course")
    has_overview = (COURSE_DIR / f"holes/hole_{hole_num:02d}_overview.png").exists()
    has_heightmap = (COURSE_DIR / "heightmap.png").exists()

    # Find nearby hole data for context
    holes_meta = metadata.get("holes", [])
    hole_meta  = next((h for h in holes_meta if h.get("hole_number") == hole_num), {})

    return render_template(
        "hole.html",
        course_name=course_name,
        hole=hole_data,
        hole_meta=hole_meta,
        prev_hole=prev_hole,
        next_hole=next_hole,
        has_overview=has_overview,
        has_heightmap=has_heightmap,
        all_holes=all_holes,
    )


@app.route("/api/holes")
def api_holes():
    """JSON API for hole list."""
    instructions = _load_instructions()
    return jsonify([{
        "number": h["hole_number"],
        "par":    h.get("par"),
        "yards":  h.get("length_yards"),
        "steps":  len(h.get("steps", [])),
    } for h in instructions])


@app.route("/api/hole/<int:hole_num>")
def api_hole(hole_num: int):
    """JSON API for single hole."""
    instructions = _load_instructions()
    hole_data    = next((h for h in instructions if h["hole_number"] == hole_num), None)
    if hole_data is None:
        return jsonify({"error": "Hole not found"}), 404
    return jsonify(hole_data)


@app.route("/api/qa")
def api_qa():
    return jsonify(_load_qa())


@app.route("/images/<path:filename>")
def serve_image(filename: str):
    """Serve course output images."""
    return send_from_directory(str(COURSE_DIR), filename)


@app.route("/guide")
def full_guide():
    """Serve the pre-generated full HTML guide if available."""
    guide_path = COURSE_DIR / "build_guide.html"
    if guide_path.exists():
        return guide_path.read_text()
    abort(404)


@app.route("/qa")
def qa_report():
    """Serve the pre-generated QA report if available."""
    qa_path = COURSE_DIR / "qa_report.html"
    if qa_path.exists():
        return qa_path.read_text()
    abort(404)


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    global COURSE_DIR

    parser = argparse.ArgumentParser(
        description="Course Replicator 2K — Companion App"
    )
    parser.add_argument(
        "--course", "-c",
        required=True,
        help="Path to the course output directory (e.g. output/old-conna-golf-club)",
    )
    parser.add_argument("--host", default=config.COMPANION_HOST)
    parser.add_argument("--port", type=int, default=config.COMPANION_PORT)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    COURSE_DIR = Path(args.course).resolve()
    if not COURSE_DIR.exists():
        print(f"ERROR: Course directory not found: {COURSE_DIR}")
        sys.exit(1)

    required = ["build_instructions.json", "course_metadata.json"]
    for f in required:
        if not (COURSE_DIR / f).exists():
            print(
                f"ERROR: {f} not found in {COURSE_DIR}. "
                "Run the pipeline first: python scripts/run_pipeline.py"
            )
            sys.exit(1)

    metadata    = _load_metadata()
    course_name = metadata.get("course_name", COURSE_DIR.name)

    print(f"\n{'='*60}")
    print(f"  Course Replicator 2K — Companion App")
    print(f"  Course: {course_name}")
    print(f"  Open on tablet/second screen:")
    print(f"  http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

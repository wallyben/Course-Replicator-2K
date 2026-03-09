#!/usr/bin/env python3
"""
Manual Golf Course Digitizer — Flask development server.

Usage
-----
    python pipeline/manual_digitizer/server.py \\
        --course output/old-conna-golf-course [--port 5050]

Endpoints
---------
    GET  /                        → digitizer.html
    GET  /course                  → course metadata (bbox, name, area …)
    GET  /satellite               → satellite_mosaic.jpg
    GET  /features                → all layer FeatureCollections
    GET  /features/<layer>        → single-layer FeatureCollection
    GET  /counts                  → {layer: featureCount}
    GET  /detected                → auto-detected pipeline GeoJSON (read-only)
    POST /save_feature            → upsert single feature  {layer, feature}
    POST /save_layer              → replace entire layer   {layer, features}
    POST /delete_feature          → remove feature by id   {layer, id}
    POST /clear_layer             → empty a layer          {layer}
    POST /export                  → write layers to pipeline output files
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file

# ── Resolve project root so relative imports work when run directly ───────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from pipeline.manual_digitizer.config import (
    DEBUG,
    HOST,
    LAYERS,
    PIPELINE_OUTPUT_LAYERS,
    PORT,
)
from pipeline.manual_digitizer.export_pipeline import export_to_pipeline
from pipeline.manual_digitizer.geojson_store import GeoJSONStore
from pipeline.manual_digitizer.satellite_loader import load_course_metadata

log = logging.getLogger("digitizer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

_FRONTEND_DIR = _PROJECT_ROOT / "frontend"
app = Flask(__name__, static_folder=str(_FRONTEND_DIR), static_url_path="/static")

# ── Global state — populated in main() ───────────────────────────────────────
_output_dir:  Path       = None   # type: ignore[assignment]
_store:       GeoJSONStore = None # type: ignore[assignment]
_course_meta: dict       = {}


# ── Utility ───────────────────────────────────────────────────────────────────

def _validate_layer(layer: str) -> None:
    if layer not in LAYERS:
        abort(400, f"Unknown layer '{layer}'. Valid layers: {', '.join(LAYERS)}")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html = _FRONTEND_DIR / "digitizer.html"
    if not html.exists():
        abort(500, f"digitizer.html not found at {html}")
    return send_file(str(html))


@app.route("/course")
def course():
    return jsonify(_course_meta)


@app.route("/satellite")
def satellite():
    sat = _course_meta.get("satellite_path")
    if not sat or not Path(sat).exists():
        abort(404, "satellite_mosaic.jpg not found — run the pipeline first")
    mime = "image/jpeg" if sat.endswith(".jpg") else "image/png"
    return send_file(sat, mimetype=mime)


@app.route("/features")
def all_features():
    return jsonify(_store.get_all())


@app.route("/features/<layer>")
def layer_features(layer: str):
    _validate_layer(layer)
    return jsonify(_store.get_layer(layer))


@app.route("/counts")
def counts():
    return jsonify(_store.feature_counts())


@app.route("/detected")
def detected():
    """
    Return auto-detected pipeline GeoJSON files for use as a drawing
    starting point.  These are READ-ONLY — they do not modify the digitizer
    store until the user clicks "Load Detected".
    """
    result: dict = {}
    for layer, filename in PIPELINE_OUTPUT_LAYERS.items():
        path = _output_dir / filename
        if not path.exists():
            continue
        try:
            fc = json.loads(path.read_text(encoding="utf-8"))
            if fc.get("features"):
                # Strip pipeline-only properties that conflict with digitizer ids
                for f in fc["features"]:
                    f.setdefault("properties", {}).pop("id", None)
                result[layer] = fc
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify(result)


@app.route("/save_feature", methods=["POST"])
def save_feature():
    data = request.get_json(force=True, silent=True) or {}
    layer   = data.get("layer")
    feature = data.get("feature")
    _validate_layer(layer)
    if not feature:
        abort(400, "Missing 'feature' in request body")
    fid = _store.upsert_feature(layer, feature)
    return jsonify({"ok": True, "id": fid})


@app.route("/save_layer", methods=["POST"])
def save_layer():
    data = request.get_json(force=True, silent=True) or {}
    layer    = data.get("layer")
    features = data.get("features") or []
    _validate_layer(layer)
    count = _store.save_layer(layer, features)
    return jsonify({"ok": True, "count": count})


@app.route("/delete_feature", methods=["POST"])
def delete_feature():
    data = request.get_json(force=True, silent=True) or {}
    layer = data.get("layer")
    fid   = data.get("id")
    _validate_layer(layer)
    if not fid:
        abort(400, "Missing 'id' in request body")
    deleted = _store.delete_feature(layer, fid)
    return jsonify({"ok": True, "deleted": deleted})


@app.route("/clear_layer", methods=["POST"])
def clear_layer():
    data  = request.get_json(force=True, silent=True) or {}
    layer = data.get("layer")
    _validate_layer(layer)
    _store.clear_layer(layer)
    return jsonify({"ok": True})


@app.route("/export", methods=["POST"])
def export():
    log.info("=== EXPORT requested ===")
    summary = export_to_pipeline(_output_dir, _store)
    return jsonify({
        "ok":      True,
        "summary": summary,
        "total":   sum(summary.values()),
        "message": (
            "Features exported to pipeline output directory.\n"
            f"Run  python scripts/run_pipeline.py  again to rebuild the build pack."
        ),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _output_dir, _store, _course_meta

    parser = argparse.ArgumentParser(
        description="Manual Golf Course Digitizer — local web interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--course",
        required=True,
        metavar="DIR",
        help="Path to pipeline output directory\n(e.g.  output/old-conna-golf-course)",
    )
    parser.add_argument("--port", type=int, default=PORT,
                        help=f"Port to listen on (default {PORT})")
    parser.add_argument("--host", default=HOST,
                        help=f"Bind address (default {HOST})")
    args = parser.parse_args()

    _output_dir = Path(args.course).resolve()
    if not _output_dir.exists():
        sys.exit(f"ERROR: course directory not found: {_output_dir}")

    log.info(f"Course dir : {_output_dir}")
    _course_meta = load_course_metadata(_output_dir)
    _store       = GeoJSONStore(_output_dir)

    log.info(f"Course     : {_course_meta['name']}")
    log.info(f"Area       : {_course_meta['area_ha']:.1f} ha")
    log.info(f"Satellite  : {'✓' if _course_meta['satellite_exists'] else '✗ (not found)'}")
    log.info(f"Store      : {_output_dir / 'digitizer'}")
    log.info("")
    log.info(f"  Open  →  http://localhost:{args.port}")
    log.info("")

    app.run(host=args.host, port=args.port, debug=DEBUG, threaded=True)


if __name__ == "__main__":
    main()

"""Digitizer module configuration — layer names, styles, and endpoint mappings."""

# ── Layer definitions ─────────────────────────────────────────────────────────

LAYERS = [
    "greens",
    "fairways",
    "bunkers",
    "water",
    "rough",
    "tees",
    "trees",
    "paths",
    "holes",
]

LAYER_LABELS = {
    "greens":   "Greens",
    "fairways": "Fairways",
    "bunkers":  "Bunkers",
    "water":    "Water Hazards",
    "rough":    "Rough Zones",
    "tees":     "Tee Boxes",
    "trees":    "Tree Clusters",
    "paths":    "Cart Paths",
    "holes":    "Hole Routes",
}

# Stroke colours used both server-side (GeoJSON properties) and client-side
LAYER_COLORS = {
    "greens":   "#2ecc71",
    "fairways": "#27ae60",
    "bunkers":  "#e67e22",
    "water":    "#3498db",
    "rough":    "#8e44ad",
    "tees":     "#e74c3c",
    "trees":    "#1a5c2a",
    "paths":    "#95a5a6",
    "holes":    "#f39c12",
}

# Geometry drawn by each layer
LAYER_DRAW_TYPE = {
    "greens":   "polygon",
    "fairways": "polygon",
    "bunkers":  "polygon",
    "water":    "polygon",
    "rough":    "polygon",
    "tees":     "rectangle",
    "trees":    "polygon",
    "paths":    "polyline",
    "holes":    "polyline",
}

# Feature type string stored in GeoJSON properties
LAYER_FEATURE_TYPE = {
    "greens":   "green",
    "fairways": "fairway",
    "bunkers":  "bunker",
    "water":    "water",
    "rough":    "rough",
    "tees":     "tee",
    "trees":    "trees",
    "paths":    "path",
    "holes":    "hole",
}

# Files overwritten in the pipeline output directory on export
PIPELINE_OUTPUT_LAYERS = {
    "greens":   "greens.geojson",
    "fairways": "fairways.geojson",
    "bunkers":  "bunkers.geojson",
    "water":    "water.geojson",
    "rough":    "rough.geojson",
    "tees":     "tees.geojson",
    "trees":    "trees.geojson",
    "paths":    "paths.geojson",
    "holes":    "holes.geojson",
}

# ── Server settings ───────────────────────────────────────────────────────────
HOST  = "0.0.0.0"
PORT  = 5050    # Use 5050 to avoid clash with companion app on 5000
DEBUG = False

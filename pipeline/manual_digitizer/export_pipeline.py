"""
Export finalized digitizer GeoJSON layers to the pipeline output directory.

Overwrites the per-type GeoJSON files that downstream modules (buildpack,
translation, qa) consume.  No other pipeline files are touched.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List

from pipeline.manual_digitizer.config import (
    LAYER_FEATURE_TYPE,
    PIPELINE_OUTPUT_LAYERS,
)
from pipeline.manual_digitizer.geojson_store import GeoJSONStore

log = logging.getLogger(__name__)


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    R    = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a    = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _linestring_length_m(coords: list) -> float:
    total = 0.0
    for i in range(len(coords) - 1):
        total += _haversine_m(
            coords[i][0],     coords[i][1],
            coords[i + 1][0], coords[i + 1][1],
        )
    return total


# ── Feature enrichment ────────────────────────────────────────────────────────

def _enrich_hole(feature: dict) -> dict:
    """
    Compute distance, tee/green positions and infer par for a hole LineString.
    Properties already set by the user (hole_number, par) are preserved.
    """
    geom   = feature.get("geometry", {})
    coords = geom.get("coordinates", [])
    props  = feature.setdefault("properties", {})

    if geom.get("type") != "LineString" or len(coords) < 2:
        return feature

    tee_lon,   tee_lat   = coords[0][0],  coords[0][1]
    green_lon, green_lat = coords[-1][0], coords[-1][1]
    dist_m  = _linestring_length_m(coords)
    dist_yd = round(dist_m * 1.09361)

    props.setdefault("tee_position",   {"lon": tee_lon,   "lat": tee_lat})
    props.setdefault("green_position", {"lon": green_lon, "lat": green_lat})
    props["distance_m"]     = round(dist_m, 1)
    props["distance_yards"] = dist_yd
    props["length_m"]       = round(dist_m, 1)
    props["length_yards"]   = dist_yd
    props["routing_source"] = "manual"

    if "par" not in props:
        if dist_yd < 250:
            props["par"] = 3
        elif dist_yd > 470:
            props["par"] = 5
        else:
            props["par"] = 4

    return feature


def _stamp_features(features: List[dict], layer: str) -> List[dict]:
    """
    Add standard pipeline-compatible properties to every feature and
    sort holes by hole_number.
    """
    ftype = LAYER_FEATURE_TYPE.get(layer, layer.rstrip("s"))
    out: List[dict] = []

    for f in features:
        props = f.setdefault("properties", {})
        props.setdefault("type",   ftype)
        props.setdefault("source", "manual")
        if layer == "holes":
            f = _enrich_hole(f)
        out.append(f)

    if layer == "holes":
        out.sort(key=lambda f: f.get("properties", {}).get("hole_number", 999))

    return out


# ── Public export ─────────────────────────────────────────────────────────────

def export_to_pipeline(output_dir: Path, store: GeoJSONStore) -> Dict[str, int]:
    """
    Write every digitizer layer to the pipeline output directory.

    Returns a {layerName: featureCount} summary dict.
    """
    output_dir = Path(output_dir).resolve()
    summary: Dict[str, int] = {}

    for layer, filename in PIPELINE_OUTPUT_LAYERS.items():
        fc       = store.get_layer(layer)
        features = _stamp_features(fc.get("features", []), layer)

        dest = output_dir / filename
        dest.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
            encoding="utf-8",
        )
        summary[layer] = len(features)
        log.info(f"  Exported {len(features):3d} {layer:<9} → {filename}")

    manifest = {
        "exported_layers": summary,
        "total_features":  sum(summary.values()),
        "source":          "manual_digitizer",
    }
    (output_dir / "digitizer_export_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    log.info(
        f"Export complete — {sum(summary.values())} total features "
        f"across {len([v for v in summary.values() if v > 0])} active layers"
    )
    return summary

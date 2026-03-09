"""
Load satellite mosaic metadata from a pipeline output directory.

Returns structured metadata consumed by the Flask server and passed to
the frontend as the /course response.
"""

import json
from pathlib import Path
from typing import Optional


def load_course_metadata(output_dir: Path) -> dict:
    """
    Parse boundary.json (required) and vision_summary.json (optional)
    from a pipeline output directory.

    Returns
    -------
    dict with keys:
        name             str   — course display name
        bbox             list  — [min_lon, min_lat, max_lon, max_lat] WGS84
        leaflet_bounds   list  — [[south, west], [north, east]] for L.imageOverlay
        centre           list  — [lon, lat] WGS84
        area_m2          float
        area_ha          float
        satellite_exists bool
        satellite_path   str | None  — absolute path to satellite_mosaic.jpg
        transform_params list | None — [west_lon, north_lat, lon/px, lat/px]
    """
    output_dir = Path(output_dir).resolve()

    boundary_path = output_dir / "boundary.json"
    if not boundary_path.exists():
        raise FileNotFoundError(
            f"boundary.json not found in {output_dir}.\n"
            "Run  python scripts/run_pipeline.py  first to generate course data."
        )

    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))

    bbox = boundary.get("bbox_wgs84", [])
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox_wgs84 in boundary.json: {bbox!r}")

    min_lon, min_lat, max_lon, max_lat = bbox

    # Leaflet imageOverlay bounds: [[south_lat, west_lon], [north_lat, east_lon]]
    leaflet_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    centre_raw = boundary.get("centre_wgs84")
    if centre_raw and len(centre_raw) == 2:
        centre = centre_raw          # [lon, lat]
    else:
        centre = [(min_lon + max_lon) / 2, (min_lat + max_lat) / 2]

    area_m2 = boundary.get("area_m2", 0.0)

    # Satellite mosaic — prefer .jpg, fall back to .png
    satellite_path: Optional[Path] = None
    for ext in ("satellite_mosaic.jpg", "satellite_mosaic.png"):
        p = output_dir / ext
        if p.exists():
            satellite_path = p
            break

    # Optional vision transform params
    transform_params = None
    vision_path = output_dir / "vision_summary.json"
    if vision_path.exists():
        try:
            vs = json.loads(vision_path.read_text(encoding="utf-8"))
            transform_params = vs.get("transform_params")
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "name":             boundary.get("matched_name") or boundary.get("name", "Unknown Course"),
        "bbox":             bbox,
        "leaflet_bounds":   leaflet_bounds,
        "centre":           centre,
        "area_m2":          round(area_m2, 1),
        "area_ha":          round(area_m2 / 10_000, 2),
        "satellite_exists": satellite_path is not None,
        "satellite_path":   str(satellite_path) if satellite_path else None,
        "transform_params": transform_params,
    }

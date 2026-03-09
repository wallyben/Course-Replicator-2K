"""
debug_viz.py — Debug visualization outputs.

UPGRADE 10: Advanced debug visualizations.

Generates three debug images after each pipeline run:

  vision_debug_overlay.jpg
      Satellite mosaic base with course boundary (white) and all
      detected features colour-coded as semi-transparent overlays.

  fairway_graph_debug.png
      Fairway polygons with adjacency graph edges drawn in cyan.
      Hole numbers annotated at fairway centroids.

  routing_debug.png
      Satellite mosaic with hole routing paths (tee→green lines),
      tee circles, green squares, and hole numbers annotated.

All outputs are written to output_dir.  Failures are non-fatal and
logged at DEBUG level to avoid polluting main pipeline output.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger(__name__)

# Per-feature-type BGR overlay colours for vision debug
_FEATURE_BGR: Dict[str, Tuple[int, int, int]] = {
    "fairway": (100, 220, 100),
    "green":   (0,   200,   0),
    "bunker":  (100, 210, 255),
    "water":   (220,  50,  50),
    "tee":     (255, 100, 100),
    "trees":   (0,    60,   0),
    "rough":   (80,  140,  80),
}

# 18-colour palette for per-hole routing lines
_HOLE_COLOURS: List[Tuple[int, int, int]] = [
    (255,  50,  50), ( 50, 200,  50), ( 50,  50, 255),
    (255, 200,  50), (200,  50, 255), ( 50, 200, 200),
    (255, 100,   0), (  0, 150, 255), (150, 255,   0),
    (255,   0, 150), (  0, 255, 150), (150,   0, 255),
    (200, 200,   0), (  0, 200, 200), (200,   0, 200),
    (255, 150, 150), (150, 255, 150), (150, 150, 255),
]


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_vision_debug_overlay(
    satellite_mosaic_path: Path,
    output_dir: Path,
    boundary_data: dict,
) -> Optional[Path]:
    """
    Generate vision_debug_overlay.jpg.

    Layers (bottom → top):
      1. Satellite mosaic
      2. Feature polygons (semi-transparent fills, 30% opacity)
      3. Course boundary polygon (white outline, 3px)
      4. Feature-type labels in legend strip at bottom

    Returns:
        Path to written image, or None on failure.
    """
    try:
        import cv2
        from PIL import Image
        from shapely.geometry import shape
    except ImportError:
        log.debug("debug_viz: cv2/PIL not available — skipping vision overlay")
        return None

    if not satellite_mosaic_path.exists():
        return None

    try:
        img = np.array(
            Image.open(str(satellite_mosaic_path)).convert("RGB"),
            dtype=np.uint8,
        )
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h_px, w_px = img.shape[:2]
    except Exception as e:
        log.debug(f"debug_viz: mosaic load failed: {e}")
        return None

    # Build geographic transform from bbox + image size
    bbox = boundary_data.get("bbox_wgs84", [])
    if len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = bbox
        lon_per_px = (max_lon - min_lon) / max(w_px, 1)
        lat_per_px = (max_lat - min_lat) / max(h_px, 1)

        def wgs_to_px(lon: float, lat: float) -> Tuple[int, int]:
            return (
                int((lon - min_lon) / lon_per_px),
                int((max_lat - lat) / lat_per_px),
            )

        # Draw course boundary
        poly_geom = boundary_data.get("boundary_wgs84")
        if poly_geom:
            try:
                coords = list(shape(poly_geom).exterior.coords)
                pts    = np.array(
                    [wgs_to_px(lon, lat) for lon, lat in coords],
                    dtype=np.int32,
                )
                cv2.polylines(img_bgr, [pts], True, (255, 255, 255), 3)
            except Exception as e:
                log.debug(f"debug_viz: boundary draw failed: {e}")

        # Draw detected features from GeoJSON files
        for feat_type, bgr in _FEATURE_BGR.items():
            for fname in (f"{feat_type}s.geojson", f"vision_{feat_type}s.geojson"):
                fpath = output_dir / fname
                if not fpath.exists():
                    continue
                try:
                    feats = json.loads(
                        fpath.read_text(encoding="utf-8")
                    ).get("features", [])
                    for f in feats[:60]:   # cap per-type for performance
                        geom = f.get("geometry", {})
                        gtype = geom.get("type", "")
                        coords_list = (
                            geom["coordinates"]
                            if gtype == "Polygon"
                            else [p for mp in geom.get("coordinates", [])
                                  for p in mp]
                            if gtype == "MultiPolygon"
                            else []
                        )
                        for ring in coords_list[:1]:
                            pts = np.array(
                                [wgs_to_px(c[0], c[1]) for c in ring],
                                dtype=np.int32,
                            )
                            overlay = img_bgr.copy()
                            cv2.fillPoly(overlay, [pts], bgr)
                            img_bgr = cv2.addWeighted(
                                img_bgr, 0.72, overlay, 0.28, 0
                            )
                            cv2.polylines(img_bgr, [pts], True, bgr, 1)
                    break  # only use first file that exists
                except Exception:
                    continue

    # Legend strip
    legend_h = 28
    legend   = np.zeros((legend_h, w_px, 3), dtype=np.uint8)
    x_offset = 4
    for feat_type, bgr in _FEATURE_BGR.items():
        cv2.rectangle(legend,
                      (x_offset, 4), (x_offset + 18, legend_h - 4),
                      bgr, -1)
        cv2.putText(legend, feat_type,
                    (x_offset + 22, legend_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
        x_offset += 95
    img_bgr = np.vstack([img_bgr, legend])

    out_path = output_dir / "vision_debug_overlay.jpg"
    try:
        cv2.imwrite(str(out_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        log.info(f"Debug viz: vision_debug_overlay.jpg written")
        return out_path
    except Exception as e:
        log.debug(f"debug_viz: write failed: {e}")
        return None


def generate_fairway_graph_debug(
    output_dir: Path,
    boundary_data: dict,
) -> Optional[Path]:
    """
    Generate fairway_graph_debug.png.

    Shows fairway polygons with adjacency graph edges (cyan lines) and
    connected-component numbers labelled at centroids.

    Reads fairway_graph.geojson (if present) for edge data.
    Returns Path or None.
    """
    try:
        import cv2
    except ImportError:
        return None

    bbox = boundary_data.get("bbox_wgs84", [])
    if len(bbox) != 4:
        return None

    min_lon, min_lat, max_lon, max_lat = bbox
    W, H = 1200, 900
    lon_per_px = (max_lon - min_lon) / W
    lat_per_px = (max_lat - min_lat) / H

    def wgs_to_px(lon: float, lat: float) -> Tuple[int, int]:
        return (
            int((lon - min_lon) / lon_per_px),
            int((max_lat - lat) / lat_per_px),
        )

    canvas = np.full((H, W, 3), 30, dtype=np.uint8)

    # Draw fairway polygons
    fw_path = output_dir / "fairways.geojson"
    centroids: List[Tuple[int, int]] = []
    if fw_path.exists():
        try:
            feats = json.loads(fw_path.read_text(encoding="utf-8")).get("features", [])
            from shapely.geometry import shape
            for f in feats:
                geom = f.get("geometry", {})
                if geom.get("type") not in ("Polygon", "MultiPolygon"):
                    continue
                try:
                    g = shape(geom)
                    cx, cy = wgs_to_px(g.centroid.x, g.centroid.y)
                    centroids.append((cx, cy))
                    rings = (geom["coordinates"]
                             if geom["type"] == "Polygon"
                             else [r for mp in geom["coordinates"] for r in mp])
                    for ring in rings[:1]:
                        pts = np.array(
                            [wgs_to_px(c[0], c[1]) for c in ring],
                            dtype=np.int32,
                        )
                        cv2.fillPoly(canvas, [pts], (60, 120, 60))
                        cv2.polylines(canvas, [pts], True, (100, 200, 100), 1)
                except Exception:
                    continue
        except Exception:
            pass

    # Draw adjacency edges from fairway_graph.geojson
    fg_path = output_dir / "fairway_graph.geojson"
    if fg_path.exists():
        try:
            feats = json.loads(fg_path.read_text(encoding="utf-8")).get("features", [])
            for f in feats:
                coords = f.get("geometry", {}).get("coordinates", [])
                if len(coords) >= 2:
                    p1 = wgs_to_px(coords[0][0], coords[0][1])
                    p2 = wgs_to_px(coords[-1][0], coords[-1][1])
                    cv2.line(canvas, p1, p2, (0, 255, 255), 2)
        except Exception:
            pass

    # Label centroids with index numbers
    for i, (cx, cy) in enumerate(centroids):
        cv2.circle(canvas, (cx, cy), 4, (200, 200, 200), -1)
        cv2.putText(canvas, str(i + 1), (cx + 5, cy - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 200), 1)

    cv2.putText(canvas, "FAIRWAY ADJACENCY GRAPH",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = output_dir / "fairway_graph_debug.png"
    try:
        cv2.imwrite(str(out_path), canvas)
        log.info(f"Debug viz: fairway_graph_debug.png written")
        return out_path
    except Exception:
        return None


def generate_routing_debug(
    routing_data: dict,
    boundary_data: dict,
    satellite_mosaic_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate routing_debug.png.

    Shows per-hole routing lines (tee→green), colour-coded by hole number,
    with tee circle markers and green rectangle markers.
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None

    if not satellite_mosaic_path.exists():
        return None

    try:
        img     = np.array(
            Image.open(str(satellite_mosaic_path)).convert("RGB"),
            dtype=np.uint8,
        )
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        h_px, w_px = img.shape[:2]
    except Exception:
        return None

    bbox = boundary_data.get("bbox_wgs84", [])
    if len(bbox) != 4:
        return None
    min_lon, min_lat, max_lon, max_lat = bbox

    def wgs_to_px(lon: float, lat: float) -> Tuple[int, int]:
        return (
            int((lon - min_lon) / (max_lon - min_lon) * w_px),
            int((max_lat - lat) / (max_lat - min_lat) * h_px),
        )

    holes = routing_data.get("holes", [])
    for i, hole in enumerate(holes):
        colour = _HOLE_COLOURS[i % len(_HOLE_COLOURS)]
        hn     = hole.get("hole_number", i + 1)

        # Accept both {lon,lat} dict and [lon,lat] list formats
        tee   = hole.get("tee_position")   or hole.get("tee_centroid")
        green = hole.get("green_position") or hole.get("green_centroid")
        if not tee or not green:
            continue

        if isinstance(tee, dict):
            t_pt = wgs_to_px(tee["lon"],   tee["lat"])
            g_pt = wgs_to_px(green["lon"], green["lat"])
        else:
            t_pt = wgs_to_px(tee[0],   tee[1])
            g_pt = wgs_to_px(green[0], green[1])

        cv2.line(img_bgr, t_pt, g_pt, colour, 2)
        cv2.circle(img_bgr, t_pt, 6, colour, -1)           # tee dot
        cv2.circle(img_bgr, t_pt, 7, (255, 255, 255), 1)   # tee ring
        cv2.rectangle(img_bgr,                              # green square
                      (g_pt[0] - 5, g_pt[1] - 5),
                      (g_pt[0] + 5, g_pt[1] + 5),
                      colour, -1)
        cv2.rectangle(img_bgr,
                      (g_pt[0] - 6, g_pt[1] - 6),
                      (g_pt[0] + 6, g_pt[1] + 6),
                      (255, 255, 255), 1)
        # Hole number label at midpoint
        mid = ((t_pt[0] + g_pt[0]) // 2, (t_pt[1] + g_pt[1]) // 2)
        cv2.putText(img_bgr, str(hn), mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img_bgr, str(hn), mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    cv2.putText(img_bgr, f"{len(holes)} holes routed",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    out_path = output_dir / "routing_debug.png"
    try:
        cv2.imwrite(str(out_path), img_bgr)
        log.info(f"Debug viz: routing_debug.png written ({len(holes)} holes)")
        return out_path
    except Exception:
        return None

"""
course_mask.py — Course boundary masking for all feature detections.

UPGRADE 1: Strict course boundary mask.

All vision-detected features are clipped to the golf course polygon before
any GeoJSON is written.  This eliminates false positives from:
  - Adjacent golf courses
  - Nearby towns and road markings
  - Rivers and lakes outside the boundary
  - Agricultural fields adjoining the course

Masking operates at two levels:
  1. Pixel level  — filters OpenCV contours before polygon conversion,
                    using a rasterised course mask
  2. GeoJSON level — clips Shapely polygon geometries to the boundary;
                     features partially outside are trimmed, not removed

Pipeline position:
    Boundary Detection → Course Polygon → Raster Mask
      → Feature Detection (inside mask only) → GeoJSON Clip
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

log = logging.getLogger(__name__)

# Safety margin: expand the boundary by this many metres so that features
# that straddle the boundary edge are not accidentally clipped out.
DEFAULT_BUFFER_M = 30.0

# Minimum fraction of a contour's pixel area that must overlap the course
# mask before the contour is kept.
MIN_MASK_OVERLAP_FRAC = 0.40


# ─── Public API ───────────────────────────────────────────────────────────────

def load_course_polygon(boundary_data: dict):
    """
    Load the course boundary as a Shapely polygon.

    Tries (in priority order):
      1. boundary_data["boundary_wgs84"] — actual course outline from OSM
      2. boundary_data["bbox_wgs84"]     — fallback to bounding-box rectangle

    Returns:
        Shapely Polygon, or None if neither source is available.
    """
    from shapely.geometry import shape, box

    # OSM course polygon (highest fidelity)
    poly_geom = boundary_data.get("boundary_wgs84")
    if poly_geom:
        try:
            p = shape(poly_geom)
            if not p.is_valid:
                p = p.buffer(0)
            if not p.is_empty:
                log.debug("course_mask: using OSM boundary polygon")
                return p
        except Exception as e:
            log.debug(f"course_mask: polygon parse failed ({e}), falling back to bbox")

    # Bounding-box rectangle fallback
    bbox = boundary_data.get("bbox_wgs84")
    if bbox and len(bbox) == 4:
        min_lon, min_lat, max_lon, max_lat = bbox
        log.debug("course_mask: using bbox rectangle as boundary")
        return box(min_lon, min_lat, max_lon, max_lat)

    log.warning("course_mask: no boundary polygon or bbox available — masking disabled")
    return None


def build_raster_mask(
    course_polygon,
    transform_params: tuple,
    image_shape: tuple,
    buffer_m: float = DEFAULT_BUFFER_M,
) -> Optional[np.ndarray]:
    """
    Rasterize the course polygon into a binary pixel mask.

    Args:
        course_polygon:   Shapely polygon in WGS84
        transform_params: (west_lon, north_lat, lon_per_px, lat_per_px)
        image_shape:      (height_px, width_px[, channels])
        buffer_m:         Safety margin in metres (default 30m)

    Returns:
        uint8 ndarray — 255 inside course, 0 outside.  None on failure.
    """
    if course_polygon is None:
        return None

    try:
        import cv2
        from shapely.ops import transform as shp_transform
        from pyproj import Transformer

        west, north, lon_per_px, lat_per_px = transform_params
        h_px, w_px = image_shape[:2]

        poly = course_polygon

        # Expand boundary by buffer_m for safety margin
        if buffer_m > 0:
            try:
                t_to_itm = Transformer.from_crs("EPSG:4326", "EPSG:2157", always_xy=True)
                t_to_wgs = Transformer.from_crs("EPSG:2157", "EPSG:4326", always_xy=True)
                poly_itm  = shp_transform(t_to_itm.transform, course_polygon)
                poly_itm  = poly_itm.buffer(buffer_m)
                poly      = shp_transform(t_to_wgs.transform, poly_itm)
            except Exception as e:
                log.debug(f"course_mask: buffer expansion failed ({e})")

        def _wgs_to_px(lon: float, lat: float) -> Tuple[int, int]:
            return (
                int((lon  - west)  / lon_per_px),
                int((north - lat)  / lat_per_px),
            )

        # Rasterize exterior ring
        ext_pts = np.array(
            [_wgs_to_px(lon, lat) for lon, lat in poly.exterior.coords],
            dtype=np.int32,
        )
        mask = np.zeros((h_px, w_px), dtype=np.uint8)
        cv2.fillPoly(mask, [ext_pts], 255)

        # Subtract interior rings (holes in the polygon)
        for interior in poly.interiors:
            hole_pts = np.array(
                [_wgs_to_px(lon, lat) for lon, lat in interior.coords],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [hole_pts], 0)

        inside_pct = 100.0 * float(np.mean(mask > 0))
        log.info(f"Course mask built: {inside_pct:.1f}% of image is within boundary")
        return mask

    except Exception as e:
        log.warning(f"Course mask build failed: {e}")
        return None


def apply_mask_to_contours(
    contours: list,
    course_mask: np.ndarray,
    min_overlap: float = MIN_MASK_OVERLAP_FRAC,
) -> list:
    """
    Filter OpenCV contours to keep only those substantially inside the mask.

    A contour is kept when:
        (pixels_inside_mask / contour_area) >= min_overlap

    Args:
        contours:    List of OpenCV contours
        course_mask: Binary uint8 mask (255 = inside course)
        min_overlap: Minimum overlap fraction (default 0.40)

    Returns:
        Filtered contour list.
    """
    if course_mask is None or len(contours) == 0:
        return contours

    try:
        import cv2

        kept   = []
        h, w   = course_mask.shape[:2]
        single = np.zeros((h, w), dtype=np.uint8)

        for c in contours:
            single[:] = 0
            cv2.drawContours(single, [c], -1, 255, thickness=cv2.FILLED)

            c_area = float(cv2.countNonZero(single))
            if c_area < 1:
                continue

            overlap = float(cv2.countNonZero(
                cv2.bitwise_and(single, course_mask)
            ))
            if overlap / c_area >= min_overlap:
                kept.append(c)

        removed = len(contours) - len(kept)
        if removed > 0:
            log.debug(
                f"course_mask: {removed}/{len(contours)} contours removed "
                f"(outside boundary)"
            )
        return kept

    except Exception as e:
        log.warning(f"apply_mask_to_contours failed: {e}")
        return contours


def generate_course_mask_debug(
    course_mask: np.ndarray,
    satellite_mosaic_path: Path,
    output_dir: Path,
) -> Optional[Path]:
    """
    Generate course_mask_debug.png — satellite mosaic with course boundary
    mask overlaid as a semi-transparent green fill.

    Writes to output_dir/course_mask_debug.png.
    Returns Path on success, None on failure.
    """
    if course_mask is None:
        return None

    try:
        import cv2
        from PIL import Image

        mosaic_path = Path(satellite_mosaic_path)
        if not mosaic_path.exists():
            return None

        img = np.array(Image.open(str(mosaic_path)).convert("RGB"), dtype=np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Resize mask if image dimensions differ (tiles may vary)
        h_img, w_img = img_bgr.shape[:2]
        h_msk, w_msk = course_mask.shape[:2]
        if (h_msk, w_msk) != (h_img, w_img):
            mask_resized = cv2.resize(course_mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = course_mask

        # Semi-transparent green overlay inside mask
        overlay = img_bgr.copy()
        overlay[mask_resized > 0] = (
            overlay[mask_resized > 0] * 0.6 +
            np.array([0, 120, 0], dtype=np.float32) * 0.4
        ).astype(np.uint8)

        # Hard boundary contour (white, 2px)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        # Label
        cv2.putText(overlay, "COURSE MASK",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        inside_pct = 100.0 * float(np.mean(mask_resized > 0))
        cv2.putText(overlay, f"{inside_pct:.1f}% of image in boundary",
                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        out_path = Path(output_dir) / "course_mask_debug.png"
        cv2.imwrite(str(out_path), overlay)
        log.info(f"Debug: course_mask_debug.png written ({inside_pct:.1f}% coverage)")
        return out_path

    except Exception as e:
        log.debug(f"course_mask_debug generation failed: {e}")
        return None


def clip_geojson_to_boundary(
    features: list,
    course_polygon,
) -> list:
    """
    GeoJSON-level boundary clip.

    For every feature:
      - If it does not intersect the boundary → removed
      - If it partially overlaps → clipped to the intersection

    This is the final clean-up layer applied after pixel-level filtering.

    Args:
        features:       List of GeoJSON Feature dicts
        course_polygon: Shapely polygon (WGS84)

    Returns:
        Clipped feature list.
    """
    if course_polygon is None:
        return features

    from shapely.geometry import shape, mapping

    clipped: List[dict] = []
    for f in features:
        if not f.get("geometry"):
            continue
        try:
            geom = shape(f["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not course_polygon.intersects(geom):
                continue
            inter = course_polygon.intersection(geom)
            if inter.is_empty:
                continue
            fc = dict(f)
            fc["geometry"] = mapping(inter)
            clipped.append(fc)
        except Exception:
            # If clip fails, keep the original feature
            clipped.append(f)

    removed = len(features) - len(clipped)
    if removed > 0:
        log.info(
            f"Boundary clip: {removed} out-of-bounds features removed "
            f"({len(clipped)} retained)"
        )
    return clipped

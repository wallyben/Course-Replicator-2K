#!/usr/bin/env python3
"""
run_pipeline.py — Main CLI entry point for Course Replicator 2K (Version 2).

Usage:
    python scripts/run_pipeline.py "Old Conna Golf Club"
    python scripts/run_pipeline.py "Ballybunion Golf Club" --type links
    python scripts/run_pipeline.py "Portmarnock Golf Club" --scorecard scorecard.json
    python scripts/run_pipeline.py --bbox -6.15,53.10,-6.10,53.15 "Custom Course"

After running, start the companion app:
    python companion/app.py --course output/old-conna-golf-club

V2 enhancements (transparent to CLI):
  FIX 1: Mapzen Terrarium tiles as reliable global elevation fallback
  FIX 2: DEM integrity validation before terrain processing
  FIX 3: Automatic bbox area refinement 40-200 ha
  FIX 4: Corrected EU-DEM LAEA tile naming with 3x retry
  UPGRADE 1: Irish LiDAR acquisition module
  UPGRADE 2: OSM per-type GeoJSON extraction
  UPGRADE 3: Satellite vision feature detection
  UPGRADE 4: Course routing reconstruction
  PART 3: NaN-safe terrain with Gaussian smoothing
  PART 4: Enhanced build pack with PDF blueprints + HTML report
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_pipeline")


def slugify(name: str) -> str:
    """Convert course name to filesystem-safe slug."""
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")[:60]


# ─── Cache validation helpers ─────────────────────────────────────────────────

_MIN_CACHE_BYTES = getattr(config, "INVALID_CACHE_MIN_BYTES", 10)


def _is_valid_cache(path: Path) -> bool:
    """
    Return True if the file exists and is large enough to be non-empty content.
    Does NOT parse the file — use _geojson_has_features for deeper checks.
    """
    if not path.exists():
        return False
    try:
        return path.stat().st_size >= _MIN_CACHE_BYTES
    except OSError:
        return False


def _geojson_has_features(path: Path) -> bool:
    """
    Return True if path is a valid GeoJSON FeatureCollection with ≥1 feature.
    Returns False if file is missing, empty, malformed, or has zero features.
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return (
            isinstance(data, dict) and
            data.get("type") == "FeatureCollection" and
            len(data.get("features", [])) > 0
        )
    except Exception:
        return False


def _normalise_routing_holes(routing_holes: list) -> list:
    """
    Convert routing.py hole dicts to the feature.py hole dict format that
    translation.py / qa.py / companion UI expect.

    Routing format:   tee_position, green_position, distance_m, distance_yards
    Features format:  tee_centroid, green_centroid, length_m, length_yards
    """
    normalised = []
    for h in routing_holes:
        tee_pos   = h.get("tee_position")   or {}
        grn_pos   = h.get("green_position") or {}
        tee_lon   = tee_pos.get("lon")
        tee_lat   = tee_pos.get("lat")
        grn_lon   = grn_pos.get("lon")
        grn_lat   = grn_pos.get("lat")

        # distance_m / distance_yards are the routing field names
        dist_m  = h.get("distance_m")    or h.get("length_m")
        dist_yd = h.get("distance_yards") or h.get("length_yards")

        # If yards were stored but metres missing (or vice versa), compute
        if dist_m is None and dist_yd is not None:
            dist_m = dist_yd * 0.9144
        if dist_yd is None and dist_m is not None:
            dist_yd = dist_m * 1.09361

        normalised.append({
            "hole_number":    h["hole_number"],
            "par":            h.get("par"),
            "handicap":       h.get("handicap"),
            "tee_centroid":   [tee_lon, tee_lat] if tee_lon is not None else None,
            "green_centroid": [grn_lon, grn_lat] if grn_lon is not None else None,
            "length_m":       round(dist_m,  1) if dist_m  is not None else None,
            "length_yards":   round(dist_yd, 0) if dist_yd is not None else None,
            "routing_source": h.get("routing_source", "inferred"),
            "bunkers":        h.get("bunkers", []),
            "fairways":       h.get("fairways", []),
        })
    return normalised


def main():
    parser = argparse.ArgumentParser(
        description="Course Replicator 2K — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py "Old Conna Golf Club"
  python scripts/run_pipeline.py "Ballybunion Golf Club" --type links
  python scripts/run_pipeline.py --bbox -6.15,53.10,-6.10,53.15 "My Course"
  python scripts/run_pipeline.py "Royal County Down" --scorecard scorecard.json
        """,
    )
    parser.add_argument("course_name", help="Golf course name (as it appears on OpenStreetMap)")
    parser.add_argument(
        "--type", "-t",
        choices=list(config.COURSE_TYPE_PRESETS.keys()),
        default="links",
        help="Course type for surface/vegetation presets (default: links)",
    )
    parser.add_argument(
        "--bbox",
        help="Manual bounding box: min_lon,min_lat,max_lon,max_lat (skip OSM boundary lookup)",
    )
    parser.add_argument(
        "--scorecard",
        help="Path to optional JSON scorecard file: [{hole, par, yards, si}, ...]",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Base output directory (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-lidar",
        action="store_true",
        help="Skip LiDAR/elevation fetch (use if already downloaded)",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Skip OSM feature extraction (use if already extracted)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Output directory ─────────────────────────────────────────────────────
    slug       = slugify(args.course_name)
    output_dir = Path(args.output_dir) / slug
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"{'='*60}")
    log.info(f"  Course Replicator 2K")
    log.info(f"  Course : {args.course_name}")
    log.info(f"  Type   : {args.type}")
    log.info(f"  Output : {output_dir}")
    log.info(f"{'='*60}")

    t_start = time.time()

    # ── Step 1: Boundary resolution ──────────────────────────────────────────
    log.info("\n[1/5] Resolving course boundary...")
    boundary_path = output_dir / "boundary.json"

    if boundary_path.exists():
        log.info("  Using cached boundary.")
        boundary_data = json.loads(boundary_path.read_text(encoding="utf-8"))
    else:
        from pipeline.boundary import resolve_boundary, boundary_from_bbox

        if args.bbox:
            parts = [float(x) for x in args.bbox.split(",")]
            if len(parts) != 4:
                log.error("--bbox must be 4 comma-separated values: min_lon,min_lat,max_lon,max_lat")
                sys.exit(1)
            boundary_data = boundary_from_bbox(*parts, course_name=args.course_name)
        else:
            try:
                boundary_data = resolve_boundary(args.course_name)
            except ValueError as e:
                log.error(f"Boundary resolution failed: {e}")
                log.error(
                    "\nTip: Try using --bbox to manually specify the bounding box:\n"
                    "  python scripts/run_pipeline.py 'Course Name' "
                    "--bbox min_lon,min_lat,max_lon,max_lat\n"
                    "  (get coordinates from openstreetmap.org)"
                )
                sys.exit(1)

        boundary_path.write_text(json.dumps(boundary_data, indent=2), encoding="utf-8")
        log.info(f"  Matched: {boundary_data['matched_name']!r} "
                 f"(confidence: {boundary_data['confidence']})")
        log.info(f"  Area: {boundary_data['area_m2']/10000:.1f} hectares")

    # ── Step 2: Elevation / LiDAR ────────────────────────────────────────────
    log.info("\n[2/5] Acquiring elevation data...")
    dtm_path = output_dir / "dtm.tif"

    if args.skip_lidar and dtm_path.exists():
        log.info("  Skipping LiDAR (--skip-lidar). Using existing dtm.tif.")
        lidar_coverage = "cached"
    elif dtm_path.exists():
        log.info("  Using cached DTM.")
        lidar_coverage = "cached"
    else:
        # V2 UPGRADE 1: Try Irish LiDAR first if course is in Ireland
        lidar_coverage = None
        bbox_wgs84 = boundary_data["bbox_wgs84"]

        try:
            from pipeline.lidar_ireland import acquire_irish_lidar, is_ireland
            if is_ireland(bbox_wgs84):
                log.info("  V2: Attempting Irish National LiDAR Programme...")
                dtm_path_irl, irl_label = acquire_irish_lidar(bbox_wgs84, output_dir)
                if dtm_path_irl and dtm_path_irl.exists():
                    dtm_path = dtm_path_irl
                    lidar_coverage = irl_label
                    log.info(f"  Irish LiDAR acquired: {irl_label}")
        except Exception as e:
            log.warning(f"  Irish LiDAR module error (falling back): {e}")

        # Standard fallback chain (includes Mapzen — FIX 1)
        if not lidar_coverage:
            from pipeline.lidar import acquire_elevation
            try:
                dtm_path, lidar_coverage = acquire_elevation(boundary_data, output_dir)
                log.info(f"  Coverage: {lidar_coverage}")
            except RuntimeError as e:
                log.error(f"Elevation acquisition failed: {e}")
                log.error(
                    "\nThe pipeline cannot continue without elevation data. "
                    "Check your internet connection and try again."
                )
                sys.exit(1)

        # V2 FIX 2: Validate DEM integrity explicitly
        try:
            from pipeline.lidar import validate_dem_integrity
            validate_dem_integrity(dtm_path)
            log.info("  DEM integrity check passed.")
        except RuntimeError as e:
            log.error(f"DEM validation failed: {e}")
            sys.exit(1)

    # ── Step 3: Terrain processing ───────────────────────────────────────────
    log.info("\n[3/5] Processing terrain...")
    terrain_stats_path = output_dir / "terrain_stats.json"

    if terrain_stats_path.exists():
        log.info("  Using cached terrain stats.")
        terrain_stats = json.loads(terrain_stats_path.read_text(encoding="utf-8"))
    else:
        from pipeline.terrain import process_terrain
        try:
            terrain_stats = process_terrain(dtm_path, boundary_data, output_dir)
            log.info(f"  Elevation range: {terrain_stats['z_min_m']:.1f}m – {terrain_stats['z_max_m']:.1f}m")
            log.info(f"  Slope: mean {terrain_stats['slope_mean_deg']:.1f}°, max {terrain_stats['slope_max_deg']:.1f}°")
        except Exception as e:
            log.error(f"Terrain processing failed: {e}")
            log.exception(e)
            sys.exit(1)

    # ── Step 4: Golf feature extraction ──────────────────────────────────────
    log.info("\n[4/5] Extracting golf features from OSM...")
    features_path = output_dir / "features.geojson"

    if args.skip_features and features_path.exists():
        log.info("  Skipping feature extraction (--skip-features).")
        # Use osm_holes_metadata.json (list format) — NOT holes_metadata.json
        # which is the routing dict written by routing.py.
        osm_holes_path = output_dir / "osm_holes_metadata.json"
        _osm_holes = []
        if osm_holes_path.exists() and _is_valid_cache(osm_holes_path):
            try:
                _osm_holes = json.loads(osm_holes_path.read_text(encoding="utf-8"))
                if not isinstance(_osm_holes, list):
                    _osm_holes = []
            except Exception:
                pass
        features_data = {
            "features_geojson": str(features_path),
            "holes_metadata":   str(osm_holes_path),
            "feature_map_path": str(output_dir / "feature_map.png"),
            "holes":            _osm_holes,
            "hole_count":       len(_osm_holes),
            "feature_counts":   {},
            "confidence_summary": {},
        }
    else:
        from pipeline.features import extract_features
        try:
            features_data = extract_features(boundary_data, output_dir)
            log.info(f"  Holes: {features_data['hole_count']}")
            log.info(f"  Features: {features_data['feature_counts']}")
        except Exception as e:
            log.error(f"Feature extraction failed: {e}")
            log.exception(e)
            sys.exit(1)

    # ── V2: OSM per-type feature extraction (UPGRADE 2) ─────────────────────
    log.info("\n[V2] Extracting per-type OSM golf features...")
    osm_summary = {}
    try:
        from pipeline.osm_features import extract_osm_features
        osm_summary = extract_osm_features(boundary_data, output_dir)
        log.info(f"  OSM features: {osm_summary.get('counts', {})}")
    except Exception as e:
        log.warning(f"  OSM feature extraction failed (non-critical): {e}")

    # ── V2: Satellite vision feature detection (UPGRADE 3) ───────────────────
    log.info("\n[V2] Running satellite vision feature detection...")
    vision_summary = {}
    try:
        from pipeline.vision_extract import (
            detect_features,
            merge_vision_with_osm,
            refilter_water_strict,
        )
        vision_summary = detect_features(
            boundary_data["bbox_wgs84"],
            output_dir,
            zoom=16,
            boundary_data=boundary_data,   # UPGRADE 1: course boundary mask
        )
        if vision_summary.get("detections"):
            merge_stats = merge_vision_with_osm(vision_summary, output_dir)
            log.info(f"  Vision detections merged: {merge_stats}")

        # ── Multi-stage validation pass ──────────────────────────────────────
        # If water is severely over-detected (>25 bodies) apply a strict
        # post-hoc filter without re-downloading tiles.  Bounded to 1 pass.
        _det   = vision_summary.get("detections", {})
        _water = _det.get("water", {}).get("count", 0)
        _green = _det.get("green", {}).get("count", 0)
        if _water > 25:
            log.info(
                f"\n[V2] Multi-stage pass 2: water over-detected ({_water}) — "
                f"applying strict post-filter..."
            )
            try:
                _new_water = refilter_water_strict(output_dir)
                log.info(f"  Water refiltered: {_water} → {_new_water}")
                # Update vision_summary so QA and validation see correct count
                if "water" in _det:
                    _det["water"]["count"] = _new_water
                vision_summary["detections"] = _det
                # Update class_filter_debug.json pass 2 entry
                try:
                    import json as _json
                    _cfd_path = output_dir / "class_filter_debug.json"
                    _cfd      = _json.loads(_cfd_path.read_text(encoding="utf-8"))
                    _cfd.setdefault("pass2_water_refilter", {})["water"] = {
                        "before": _water,
                        "after":  _new_water,
                    }
                    _cfd_path.write_text(_json.dumps(_cfd, indent=2), encoding="utf-8")
                except Exception:
                    pass
            except Exception as we:
                log.warning(f"  Water refilter failed (non-critical): {we}")

        # UPGRADE 9: Feature count validation
        try:
            from pipeline.feature_validation import (
                validate_feature_counts,
                write_validation_report,
            )
            _feat_counts = {
                k: v.get("count", 0)
                for k, v in vision_summary.get("detections", {}).items()
            }
            _val_results = validate_feature_counts(_feat_counts)
            write_validation_report(_feat_counts, _val_results, output_dir)
            log.info(f"  Feature validation: {_val_results}")
        except Exception as ve:
            log.debug(f"Feature validation failed (non-critical): {ve}")

        # Boundary fitness check: verify the boundary covers the right course
        try:
            from pipeline.boundary import verify_boundary_fit
            _det_counts = {
                k: v.get("count", 0)
                for k, v in vision_summary.get("detections", {}).items()
            }
            _bfit_ok, _bfit_reason = verify_boundary_fit(boundary_data, _det_counts)
            if _bfit_ok:
                log.info(f"  Boundary verification: {_bfit_reason}")
            else:
                log.warning(f"  Boundary verification FAILED: {_bfit_reason}")
                log.warning(
                    "  TIP: If the wrong course is being reconstructed, delete "
                    "boundary.json and re-run (Nominatim will re-geocode)."
                )
        except Exception as bve:
            log.debug(f"Boundary verification failed (non-critical): {bve}")
    except Exception as e:
        log.warning(f"  Vision extraction failed (non-critical): {e}")

    # ── V3: Satellite-based tee detection (UPGRADE 5) ────────────────────────
    log.info("\n[V3] Detecting tee boxes from satellite imagery...")
    tees_path = output_dir / "tees.geojson"
    # A valid cache must exist, have content, and contain a FeatureCollection
    # with at least one feature.  An empty FeatureCollection (from osm_features
    # writing zero OSM tees) is NOT a valid cache — we regenerate it.
    _tees_valid_cache = (
        _is_valid_cache(tees_path) and
        _geojson_has_features(tees_path)
    )
    if not getattr(config, "ENABLE_TEE_DETECTION", True):
        log.info("  Tee detection disabled via config.")
    elif _tees_valid_cache:
        log.info(f"  Using cached tees.geojson ({tees_path.stat().st_size} bytes)")
    else:
        if tees_path.exists() and not _tees_valid_cache:
            log.info("  tees.geojson exists but is empty/invalid — regenerating.")
        try:
            from pipeline.tee_detection import detect_tees
            mosaic_path = output_dir / "satellite_mosaic.jpg"
            tee_result  = detect_tees(
                bbox_wgs84=boundary_data["bbox_wgs84"],
                output_dir=output_dir,
                satellite_mosaic_path=mosaic_path if mosaic_path.exists() else None,
            )
            n_tees = tee_result.get("tee_count", 0)
            log.info(f"  Tees detected: {n_tees} — written to {tees_path.name}")
        except Exception as e:
            log.warning(f"  Tee detection failed (non-critical): {e}")

    # ── V3: Optional ML vision refinement (UPGRADE 6) ────────────────────────
    if config.ENABLE_ML_VISION:
        log.info("\n[V3] Running ML vision refinement...")
        try:
            from pipeline.ml_vision import refine_vision_detections
            ml_stats = refine_vision_detections(
                output_dir=output_dir,
                satellite_mosaic_path=output_dir / "satellite_mosaic.jpg",
            )
            if ml_stats:
                log.info(f"  ML refinement stats: {ml_stats}")
        except Exception as e:
            log.warning(f"  ML vision refinement failed (non-critical): {e}")

    # ── V2: Course routing reconstruction (UPGRADE 4) ─────────────────────────
    log.info("\n[V2] Reconstructing course routing...")
    routing_data = {}
    routing_path = output_dir / "holes.geojson"
    if _is_valid_cache(routing_path) and _geojson_has_features(routing_path):
        log.info(f"  Using cached routing ({routing_path.name})")
        try:
            # Read actual hole properties from holes.geojson — do NOT read
            # holes_metadata.json as a list (it is the routing dict, not holes).
            _cached_feats = json.loads(routing_path.read_text(encoding="utf-8")).get("features", [])
            # Keep only routing LineString/Point features (exclude fairway corridors)
            _cached_holes = [
                f["properties"] for f in _cached_feats
                if f.get("properties", {}).get("hole_number")
                   and f.get("geometry", {}).get("type") in ("LineString", "Point", None)
                   and f.get("properties", {}).get("type") != "fairway_corridor"
            ]
            routing_data = {
                "hole_count":  len(_cached_holes),
                "holes":       _cached_holes,
                "holes_path":  str(routing_path),
            }
            log.info(f"  Cached routing: {len(_cached_holes)} holes loaded")

            # CRITICAL BUG FIX: holes_metadata.json may be missing or 0KB on
            # cached runs (routing.py was not called → file never written).
            # Regenerate it from the cached hole list so QA + translation have
            # a valid metadata file regardless of whether routing re-ran.
            _meta_path = output_dir / "holes_metadata.json"
            if not _is_valid_cache(_meta_path):
                _meta_payload = json.dumps({
                    "hole_count":        len(_cached_holes),
                    "routing_method":    "inferred",
                    "source":            "vision + osm",
                    "has_inferred_tees": True,
                    "cached":            True,
                }, indent=2)
                _meta_path.write_text(_meta_payload, encoding="utf-8")
                log.info(
                    f"  Routing cache: holes_metadata.json regenerated — "
                    f"{len(_cached_holes)} holes, {_meta_path.stat().st_size} bytes"
                )

        except Exception as e:
            log.warning(f"  Cached routing read failed ({e}) — re-running routing")
            routing_data = {}

    if not routing_data:
        try:
            from pipeline.routing import reconstruct_routing
            routing_data = reconstruct_routing(
                features_data=features_data,
                output_dir=output_dir,
                osm_features_dir=output_dir,
            )
            log.info(f"  Routing: {routing_data.get('hole_count', 0)} holes → holes.geojson")
        except Exception as e:
            log.warning(f"  Routing reconstruction failed (non-critical): {e}")
            routing_data = {
                "hole_count": features_data.get("hole_count", 0),
                "holes":      features_data.get("holes", []),
            }

    # ── Inject routing holes into features_data for translation ──────────────
    # features_data["holes"] comes from OSM only and is empty for most courses.
    # Routing generates 18 holes; we normalise and merge them so that
    # generate_build_pack() produces real yardage / hole counts.
    routing_holes = routing_data.get("holes", [])
    if routing_holes and not features_data.get("holes"):
        normalised = _normalise_routing_holes(routing_holes)
        features_data["holes"]      = normalised
        features_data["hole_count"] = len(normalised)
        log.info(
            f"Translation: derived {len(normalised)} holes from routing "
            f"(features_data was empty)"
        )
    elif routing_holes and len(routing_holes) > len(features_data.get("holes", [])):
        # Routing produced more holes than OSM — prefer routing
        normalised = _normalise_routing_holes(routing_holes)
        features_data["holes"]      = normalised
        features_data["hole_count"] = len(normalised)
        log.info(
            f"Translation: replaced {features_data.get('hole_count', 0)} OSM holes "
            f"with {len(normalised)} routing holes (routing is more complete)"
        )

    # ── Step 5: 2K translation + build pack ─────────────────────────────────
    log.info("\n[5/5] Generating 2K build pack...")

    scorecard = None
    if args.scorecard:
        sc_path = Path(args.scorecard)
        if sc_path.exists():
            scorecard = json.loads(sc_path.read_text(encoding="utf-8"))
            log.info(f"  Loaded scorecard: {len(scorecard)} holes")
        else:
            log.warning(f"  Scorecard file not found: {sc_path}")

    from pipeline.translation import generate_build_pack
    try:
        build_pack = generate_build_pack(
            boundary_data=boundary_data,
            terrain_stats=terrain_stats,
            features_data=features_data,
            output_dir=output_dir,
            course_type=args.type,
            scorecard=scorecard,
        )
        log.info(f"  Total holes: {build_pack['total_holes']}")
        log.info(f"  Total yards: {int(build_pack['total_yards']):,}")
        log.info(f"  Fidelity:    {build_pack['fidelity_estimate'].get('overall_score', '?')}%")
    except Exception as e:
        log.error(f"Build pack generation failed: {e}")
        log.exception(e)
        sys.exit(1)

    # ── V2: Enhanced build pack (PART 4) ─────────────────────────────────────
    log.info("\n[V2] Generating enhanced build pack...")
    try:
        from pipeline.buildpack import generate_enhanced_buildpack
        enhanced_pack = generate_enhanced_buildpack(
            boundary_data=boundary_data,
            terrain_stats=terrain_stats,
            features_data=features_data,
            routing_data=routing_data,
            output_dir=output_dir,
            course_type=args.type,
            osm_features_dir=output_dir,
        )
        log.info("  Enhanced outputs:")
        for key, path in enhanced_pack.get("outputs", {}).items():
            if path:
                log.info(f"    {key}: {Path(path).name}")
    except Exception as e:
        log.warning(f"  Enhanced build pack failed (non-critical): {e}")

    # ── UPGRADE 10: Debug visualizations ─────────────────────────────────────
    log.info("\n[V3] Generating debug visualizations...")
    try:
        from pipeline.debug_viz import (
            generate_vision_debug_overlay,
            generate_fairway_graph_debug,
            generate_routing_debug,
        )
        generate_vision_debug_overlay(
            output_dir / "satellite_mosaic.jpg",
            output_dir,
            boundary_data,
        )
        generate_fairway_graph_debug(output_dir, boundary_data)
        generate_routing_debug(
            routing_data,
            boundary_data,
            output_dir / "satellite_mosaic.jpg",
            output_dir,
        )
    except Exception as dve:
        log.debug(f"Debug visualizations failed (non-critical): {dve}")

    # course_mask_debug.png — boundary mask overlaid on satellite mosaic
    try:
        from pipeline.course_mask import (
            load_course_polygon,
            build_raster_mask,
            generate_course_mask_debug,
        )
        from pipeline.vision_extract import _download_satellite_mosaic
        mosaic_path = output_dir / "satellite_mosaic.jpg"
        if mosaic_path.exists():
            # Rebuild mask from boundary_data (no network call needed)
            _cp = load_course_polygon(boundary_data)
            if _cp is not None:
                # Approximate transform from bbox (no actual tile download)
                _bbox = boundary_data["bbox_wgs84"]
                import json as _j
                _vis_sum_path = output_dir / "vision_summary.json"
                _iw, _ih = 1024, 1024
                if _vis_sum_path.exists():
                    try:
                        _vs = _j.loads(_vis_sum_path.read_text())
                        _iw, _ih = _vs.get("image_size", [_iw, _ih])
                    except Exception:
                        pass
                _tp = (
                    _bbox[0],
                    _bbox[3],
                    (_bbox[2] - _bbox[0]) / max(_iw, 1),
                    (_bbox[3] - _bbox[1]) / max(_ih, 1),
                )
                _cm = build_raster_mask(_cp, _tp, (_ih, _iw))
                generate_course_mask_debug(_cm, mosaic_path, output_dir)
    except Exception as cme:
        log.debug(f"course_mask_debug generation failed (non-critical): {cme}")

    # ── QA ───────────────────────────────────────────────────────────────────
    log.info("\nRunning QA checks...")
    from pipeline.qa import run_qa
    try:
        qa_report = run_qa(
            boundary_data=boundary_data,
            terrain_stats=terrain_stats,
            features_data=features_data,
            build_pack=build_pack,
            output_dir=output_dir,
            expected_scorecard=scorecard,
            routing_data=routing_data,
        )
        log.info(f"  QA status: {qa_report['overall_status']}")
        if qa_report["warnings"]:
            for w in qa_report["warnings"]:
                log.warning(f"  ⚠ {w}")
        if qa_report["errors"]:
            for e in qa_report["errors"]:
                log.error(f"  ✗ {e}")
    except Exception as e:
        log.warning(f"QA check failed (non-critical): {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    log.info(f"\n{'='*60}")
    log.info(f"  BUILD PACK COMPLETE ({elapsed:.0f}s)")
    log.info(f"  Output: {output_dir}")
    log.info(f"")
    log.info(f"  Files generated:")
    for f in sorted(output_dir.rglob("*")):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            log.info(f"    {f.relative_to(output_dir)} ({size_kb:.0f}KB)")
    log.info(f"")
    log.info(f"  Next step — start the companion app:")
    log.info(f"    python companion/app.py --course {output_dir}")
    log.info(f"  Then open http://localhost:{config.COMPANION_PORT} on a tablet/second screen.")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
run_pipeline.py — Main CLI entry point for Course Replicator 2K.

Usage:
    python scripts/run_pipeline.py "Lahinch Golf Club"
    python scripts/run_pipeline.py "Ballybunion Golf Club" --type links
    python scripts/run_pipeline.py "Portmarnock Golf Club" --scorecard scorecard.json
    python scripts/run_pipeline.py --bbox -9.35,52.95,-9.30,53.00 "Custom Course"

After running, start the companion app:
    python companion/app.py --course output/lahinch-golf-club
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


def main():
    parser = argparse.ArgumentParser(
        description="Course Replicator 2K — Full Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py "Lahinch Golf Club"
  python scripts/run_pipeline.py "Ballybunion Golf Club" --type links
  python scripts/run_pipeline.py --bbox -9.35,52.95,-9.30,53.00 "My Course"
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
        boundary_data = json.loads(boundary_path.read_text())
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

        boundary_path.write_text(json.dumps(boundary_data, indent=2))
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

    # ── Step 3: Terrain processing ───────────────────────────────────────────
    log.info("\n[3/5] Processing terrain...")
    terrain_stats_path = output_dir / "terrain_stats.json"

    if terrain_stats_path.exists():
        log.info("  Using cached terrain stats.")
        terrain_stats = json.loads(terrain_stats_path.read_text())
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
        holes_path    = output_dir / "holes_metadata.json"
        features_data = {
            "features_geojson": str(features_path),
            "holes_metadata":   str(holes_path),
            "feature_map_path": str(output_dir / "feature_map.png"),
            "holes":            json.loads(holes_path.read_text()) if holes_path.exists() else [],
            "hole_count":       0,
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

    # ── Step 5: 2K translation + build pack ─────────────────────────────────
    log.info("\n[5/5] Generating 2K build pack...")

    scorecard = None
    if args.scorecard:
        sc_path = Path(args.scorecard)
        if sc_path.exists():
            scorecard = json.loads(sc_path.read_text())
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

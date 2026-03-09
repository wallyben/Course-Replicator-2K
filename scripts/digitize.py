#!/usr/bin/env python3
"""
scripts/digitize.py — Convenience launcher for the Manual Golf Course Digitizer.

Usage
-----
    python scripts/digitize.py "Old Conna Golf Course"
    python scripts/digitize.py "Old Conna Golf Course" --port 5050
    python scripts/digitize.py --course output/old-conna-golf-course

The first form resolves the slug from the course name and looks for the
pipeline output directory automatically.  The --course form accepts an
explicit path (identical to calling server.py directly).

After the digitizer runs and the user clicks Export, re-run the pipeline
to regenerate the build pack with the manual features:

    python scripts/run_pipeline.py "Old Conna Golf Course" --skip-lidar

The pipeline will detect digitizer_export_manifest.json and skip vision
detection automatically, preserving all manually traced features.
"""

import argparse
import os
import re
import sys
from pathlib import Path

# Resolve project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import config


def slugify(name: str) -> str:
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")[:60]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manual Golf Course Digitizer — convenience launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/digitize.py "Old Conna Golf Course"
  python scripts/digitize.py "Ballybunion Golf Club" --port 5050
  python scripts/digitize.py --course output/old-conna-golf-course

After exporting, rebuild the pack (manual features are auto-preserved):
  python scripts/run_pipeline.py "Old Conna Golf Course" --skip-lidar
        """,
    )
    # Accept either a course name (auto-resolves slug) or an explicit --course path
    parser.add_argument(
        "course_name",
        nargs="?",
        help="Golf course name — resolves to output/<slug>/",
    )
    parser.add_argument(
        "--course",
        metavar="DIR",
        help="Explicit path to pipeline output directory",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Base output directory (default: {config.OUTPUT_DIR})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=getattr(config, "DIGITIZER_PORT", 5050),
        help="Port for the digitizer web server (default 5050)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Bind address (default 0.0.0.0)",
    )
    args = parser.parse_args()

    # Resolve course directory
    if args.course:
        course_dir = Path(args.course).resolve()
    elif args.course_name:
        slug       = slugify(args.course_name)
        course_dir = (Path(args.output_dir) / slug).resolve()
    else:
        parser.error("Provide a course name or --course <dir>")

    if not course_dir.exists():
        print(
            f"ERROR: Course directory not found: {course_dir}\n\n"
            f"Run the pipeline first:\n"
            f"  python scripts/run_pipeline.py \"{args.course_name or course_dir.name}\"\n",
            file=sys.stderr,
        )
        sys.exit(1)

    boundary_path = course_dir / "boundary.json"
    if not boundary_path.exists():
        print(
            f"ERROR: boundary.json not found in {course_dir}.\n"
            f"Run the pipeline first to generate course data.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # Check for existing digitizer export
    manifest = course_dir / "digitizer_export_manifest.json"
    if manifest.exists():
        import json
        mf = json.loads(manifest.read_text(encoding="utf-8"))
        total = mf.get("total_features", 0)
        print(f"INFO: Existing digitizer export found ({total} features).")
        print("      Continuing from previous session.\n")
    else:
        print("INFO: Starting new digitizer session.\n")

    # Delegate to the actual server
    from pipeline.manual_digitizer.server import main as server_main
    import sys as _sys

    _sys.argv = [
        "server.py",
        "--course", str(course_dir),
        "--port",   str(args.port),
        "--host",   args.host,
    ]
    server_main()


if __name__ == "__main__":
    main()

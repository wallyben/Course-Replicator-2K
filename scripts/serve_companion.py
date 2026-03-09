#!/usr/bin/env python3
"""
serve_companion.py — Quick launcher for the companion app.

Usage:
    python scripts/serve_companion.py old-conna-golf-club
    python scripts/serve_companion.py output/old-conna-golf-club --port 8080
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config

def main():
    parser = argparse.ArgumentParser(description="Start companion app")
    parser.add_argument(
        "course",
        help="Course slug (e.g. old-conna-golf-club) or full path to output directory",
    )
    parser.add_argument("--port", type=int, default=config.COMPANION_PORT)
    parser.add_argument("--host", default=config.COMPANION_HOST)
    args = parser.parse_args()

    course_path = Path(args.course)
    if not course_path.exists():
        # Try under output/
        course_path = Path(config.OUTPUT_DIR) / args.course
    if not course_path.exists():
        print(f"Error: Course directory not found: {args.course}")
        print(f"Available courses:")
        output_dir = Path(config.OUTPUT_DIR)
        if output_dir.exists():
            for d in sorted(output_dir.iterdir()):
                if d.is_dir():
                    print(f"  {d.name}")
        sys.exit(1)

    # Launch companion app
    from companion.app import app, COURSE_DIR
    import companion.app as companion_module
    companion_module.COURSE_DIR = course_path.resolve()

    print(f"\nCompanion app: http://{args.host}:{args.port}")
    print(f"Course: {course_path.resolve()}")
    print("Press Ctrl+C to stop.\n")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()

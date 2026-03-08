#!/usr/bin/env python3
"""
export_buildpack.py — Export a build pack as a portable ZIP archive.

Packages all output files into a single ZIP for easy transfer/backup.

Usage:
    python scripts/export_buildpack.py lahinch-golf-club
    python scripts/export_buildpack.py output/lahinch-golf-club --out ~/Desktop/
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


def main():
    parser = argparse.ArgumentParser(description="Export course build pack as ZIP")
    parser.add_argument("course", help="Course slug or full path to output directory")
    parser.add_argument("--out", default=".", help="Output directory for the ZIP file")
    args = parser.parse_args()

    course_path = Path(args.course)
    if not course_path.exists():
        course_path = Path(config.OUTPUT_DIR) / args.course
    if not course_path.exists():
        print(f"Error: Course directory not found: {args.course}")
        sys.exit(1)

    zip_name = f"{course_path.name}_buildpack.zip"
    zip_path = Path(args.out) / zip_name

    print(f"Packaging: {course_path}")
    print(f"Output:    {zip_path}")

    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(course_path.rglob("*")):
            if file.is_file():
                arcname = file.relative_to(course_path.parent)
                zf.write(file, arcname)
                file_count += 1

    zip_size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"Done: {file_count} files, {zip_size_mb:.1f} MB → {zip_path}")


if __name__ == "__main__":
    main()

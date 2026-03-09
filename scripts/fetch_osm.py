#!/usr/bin/env python3
"""
fetch_osm.py — Standalone OSM data fetcher and inspector.

Useful for checking OSM coverage before running the full pipeline.

Usage:
    python scripts/fetch_osm.py "Old Conna Golf Club"
    python scripts/fetch_osm.py "Ballybunion Golf Club" --save
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.boundary import resolve_boundary
from pipeline.features import _fetch_golf_features, _classify_element


def main():
    parser = argparse.ArgumentParser(description="Inspect OSM golf data for a course")
    parser.add_argument("course_name", help="Course name")
    parser.add_argument("--save", action="store_true", help="Save raw OSM data to disk")
    args = parser.parse_args()

    print(f"\nFetching boundary for: {args.course_name!r}")
    try:
        boundary = resolve_boundary(args.course_name)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Matched: {boundary['matched_name']!r}")
    print(f"OSM: {boundary['osm_type']}/{boundary['osm_id']}")
    print(f"Confidence: {boundary['confidence']}")
    print(f"Area: {boundary['area_m2']/10000:.1f} hectares")
    print(f"BBox: {boundary['bbox_wgs84']}")

    print(f"\nFetching golf features...")
    raw = _fetch_golf_features(boundary["bbox_wgs84"])
    elements = raw.get("elements", [])

    types = {}
    for el in elements:
        tags = el.get("tags", {})
        ft   = _classify_element(tags)
        if ft:
            types[ft] = types.get(ft, 0) + 1

    print(f"\nFeature summary ({len(elements)} OSM elements):")
    for ft, count in sorted(types.items()):
        print(f"  {ft:15s}: {count}")

    if not types:
        print("  (none found — course may have minimal OSM tagging)")

    if args.save:
        out = f"osm_{args.course_name.lower().replace(' ','_')}.json"
        with open(out, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"\nRaw OSM data saved: {out}")


if __name__ == "__main__":
    main()

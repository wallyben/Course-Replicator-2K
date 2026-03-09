# Course Replicator 2K

**Personal-use system for replicating Irish golf courses in PGA TOUR 2K on Xbox.**

Automated geospatial pipeline + local companion web app. No cloud. No SaaS.
Takes a real Irish golf course name as input, produces a near-finished build pack,
and guides you hole-by-hole through the in-game build via a second-screen app.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline for a course
python scripts/run_pipeline.py "Old Conna Golf Club" --type parkland

# 3. Start the companion app (on second screen / tablet beside Xbox)
python companion/app.py --course output/old-conna-golf-club

# 4. Open in browser on second screen
# http://localhost:5000
```

---

## What It Does

| Stage | What happens | Output |
|-------|-------------|--------|
| Boundary | Fetches course boundary from OpenStreetMap | `boundary.json` |
| Elevation | Downloads Irish LiDAR (INLP) or EU-DEM/SRTM fallback | `dtm.tif` |
| Terrain | Generates heightmap + slope map | `heightmap.png`, `slope_map.png` |
| Features | Extracts fairways, greens, bunkers, tees, water from OSM | `features.geojson` |
| Translation | Converts real geometry → 2K build steps + distances | `build_instructions.json` |
| QA | Validates distances, elevations, hazard presence | `qa_report.json` |
| Companion | Serves per-hole build guide locally for second-screen use | `http://localhost:5000` |

---

## Usage

### Full pipeline

```bash
python scripts/run_pipeline.py "Royal County Down Golf Club" --type links
python scripts/run_pipeline.py "Portmarnock Golf Club" --type parkland
python scripts/run_pipeline.py "Waterville Golf Links" --type links

# With manual scorecard (recommended for accuracy)
python scripts/run_pipeline.py "Old Conna Golf Club" --scorecard old_conna_scorecard.json

# Manual bounding box if OSM lookup fails
python scripts/run_pipeline.py "My Course" --bbox -6.15,53.10,-6.10,53.15
```

### Scorecard JSON format

```json
[
  {"hole": 1, "par": 4, "yards": 398, "si": 7},
  {"hole": 2, "par": 4, "yards": 481, "si": 3},
  ...
]
```

### Check OSM coverage first

```bash
python scripts/fetch_osm.py "Ballybunion Golf Club"
```

### Export build pack as ZIP

```bash
python scripts/export_buildpack.py old-conna-golf-club --out ~/Desktop/
```

---

## Course Types

| Type | Rough | Greens | Ground |
|------|-------|--------|--------|
| `links` | Thick Fescue | Fast | Firm |
| `parkland` | Thick Rough | Medium | Soft |
| `heathland` | Heather | Medium-Fast | Medium |
| `clifftop` | Thick Fescue | Fast | Firm |
| `inland` | Medium Rough | Medium | Soft |

---

## Data Sources (Priority Order)

1. **Irish National LiDAR Programme (INLP)** — 0.5m, via Tailte Éireann
2. **EU-DEM v1.1** — 25m, via Copernicus
3. **SRTM 30m** — via OpenTopography (free API key optional)
4. **Open-Elevation API** — 90m, zero-config fallback

Golf features via **OpenStreetMap Overpass API** (free, no key required).

---

## Output Directory Structure

```
output/
└── old-conna-golf-club/
    ├── boundary.json           Course boundary + OSM metadata
    ├── dtm.tif                 Raw DTM raster
    ├── dtm_clipped.tif         DTM clipped to course area
    ├── heightmap.png           16-bit greyscale heightmap reference
    ├── slope_map.png           Colour-coded slope classification
    ├── terrain_stats.json      Elevation statistics
    ├── features.geojson        All OSM golf features
    ├── holes_metadata.json     Per-hole routing + bunker data
    ├── feature_map.png         Colour-coded overhead feature map
    ├── build_instructions.json Machine-readable build steps
    ├── build_guide.html        Full standalone HTML build guide
    ├── course_metadata.json    Distances, pars, conditions
    ├── qa_report.json          QA check results
    ├── qa_report.html          QA report (human-readable)
    └── holes/
        ├── hole_01_overview.png
        ├── hole_02_overview.png
        └── ...
```

---

## Companion App

The companion app runs locally on `localhost:5000` and is designed for use on a
tablet or second monitor placed beside your Xbox.

Features:
- Per-hole overhead maps + heightmap reference
- Step-by-step build instructions with 2K-specific measurements
- Confidence indicators for each data element
- Checkbox progress tracking (persisted in browser localStorage)
- Keyboard shortcuts: `n`/`→` = next hole, `p`/`←` = prev hole, `h` = home
- Links to OpenStreetMap and Google Satellite for each hole
- QA report access

```bash
python companion/app.py --course output/old-conna-golf-club
python companion/app.py --course output/old-conna-golf-club --port 8080
```

---

## Realistic Expectations

| Scenario | Fidelity |
|----------|----------|
| Good INLP coverage + good OSM tagging | 75–85% |
| EU-DEM + good OSM tagging | 60–70% |
| SRTM + sparse OSM | 45–55% |
| Manual bbox + no OSM features | 30–40% |

**What is automated**: terrain, routing, distances, surface zones, bunker positions, step generation.

**What is always manual**: in-game sculpting, vegetation placement, course conditions, fine green undulation, visual atmosphere.

---

## Requirements

```
Python 3.11+
geopandas, shapely, pyproj, rasterio, numpy
Pillow, matplotlib, requests, flask, geopy, tqdm, pandas
```

Optional: `pdal` for native INLP point cloud processing.

---

## Legal

All data sources used are open/public:
- OpenStreetMap data: © OpenStreetMap contributors (ODbL)
- Irish National LiDAR Programme: © Tailte Éireann (Open Government Licence)
- EU-DEM: © Copernicus Programme
- SRTM: NASA (public domain)

This system is for personal, non-commercial use only.
PGA TOUR 2K is a trademark of 2K Games. This project is not affiliated with 2K.

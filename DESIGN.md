# Irish Golf Course Replicator — System Design Document
**Senior Multi-Disciplinary Golf Course Replication Council**
*Controlled system design pass. Not brainstorming. Not hype.*

---

## SECTION 1 — TRUTHFUL FEASIBILITY VERDICT

### What Is Realistically Possible

- **Automated terrain extraction**: Irish LiDAR coverage via the Irish National LiDAR Programme (INLP) / Tailte Éireann covers the majority of the island at 0.5–1m resolution. Where available, accurate DTMs can be generated for any golf course boundary.
- **Automated golf feature extraction**: OpenStreetMap has strong golf course tagging in Ireland. Fairways, greens, bunkers, tees, water hazards, and paths exist as polygon/line geometry for most well-known Irish courses.
- **Heightmap generation**: A normalised 16-bit grayscale heightmap usable as a reference image can be generated automatically.
- **Feature overlay maps**: Per-hole overhead maps with colour-coded features (fairway, green, bunker, water, tee) can be generated automatically.
- **Build instructions**: Hole-by-hole, step-by-step construction guides in 2K's tool vocabulary can be generated from the extracted data.
- **Distance and par validation**: Scorecard data (OpenStreetMap, club websites) can populate a validation matrix.
- **Companion UI**: A local web app serving side-by-side reference (heightmap + feature map + instructions) for use on a second screen beside the Xbox is fully buildable.

### What Is Not Realistically Possible

- **Direct data import into PGA TOUR 2K on Xbox**: PGA TOUR 2K21/23/25 on Xbox has no external file import capability. All terrain sculpting and feature placement happens through the in-game Course Designer using a controller. No workaround exists. No PC mod pipeline transfers to Xbox.
- **Fully automated course creation**: The final build step is human. This is a hard constraint of the platform.
- **Sub-metre bunker accuracy from OSM alone**: OSM bunker polygons are often traced from aerial imagery and carry 2–5m positional error. Shapes are approximate.
- **Vegetation match**: Individual tree species, canopy density, and rough boundary vegetation cannot be automated. Vegetation in 2K is manually placed or zone-painted.
- **100% LiDAR coverage for every Irish course**: The INLP is not fully complete. Some courses (particularly in the west) may fall back to EU-DEM or SRTM (30m resolution), significantly reducing terrain precision.
- **Legal data access for high-resolution aerial imagery at scale**: Google/Bing satellite tiles cannot be legally bulk-downloaded and processed as rasters. They serve as visual reference only.

### Best Achievable End-State

A single determined builder can use this system to:
1. Input a course name
2. Receive an automated build pack within ~15 minutes (data fetch + processing)
3. Sit beside an Xbox with a second screen running the companion app
4. Build a recognisable, accurately-routed, correctly-distanced, terrain-faithful version of any covered Irish golf course in PGA TOUR 2K with dramatically reduced manual guesswork

**Target fidelity ceiling**: 75–85% geometric accuracy on courses with good LiDAR + OSM coverage. 50–65% on poorly-covered courses. Visual/atmospheric fidelity is always manual.

---

## SECTION 2 — TARGET WORKFLOW

```
INPUT: Course name (e.g. "Old Conna Golf Club")
       │
       ▼
[1] BOUNDARY RESOLUTION
    - Overpass API → OSM golf course boundary polygon
    - Fallback: manual bounding box entry
       │
       ▼
[2] DATA ACQUISITION (parallel)
    ├── OSM golf features (Overpass API)
    ├── LiDAR tiles (Tailte Éireann INLP → fallback EU-DEM → fallback SRTM)
    └── Scorecard data (OpenStreetMap relation tags → fallback manual)
       │
       ▼
[3] TERRAIN PROCESSING
    - Point cloud → DTM (Digital Terrain Model)
    - DTM → Heightmap (normalised 16-bit PNG, 2K-scaled)
    - Slope map extraction
    - Terrain region classification (flat/gentle/steep)
       │
       ▼
[4] GOLF FEATURE EXTRACTION
    - Fairway polygons + confidence score
    - Green polygons + confidence score
    - Bunker polygons + confidence score
    - Tee polygons + confidence score
    - Water hazard polygons + confidence score
    - Hole routing (ordered tee→green per hole)
    - Path/cart track geometry
       │
       ▼
[5] 2K TRANSLATION
    - Real-world metres → 2K grid units (scaled)
    - Terrain height range → 2K slider range
    - Feature polygons → 2K surface paint regions
    - Hazard polygons → 2K hazard placement guides
    - Vegetation budget per hole
    - Par/distance/stroke assignment
       │
       ▼
[6] BUILD PACK GENERATION
    - heightmap.png (16-bit grayscale reference)
    - feature_map.png (colour-coded overhead)
    - hole_XX_overview.png (per-hole annotated maps)
    - slope_map.png (shaded relief)
    - build_instructions.json (machine-readable)
    - build_guide.html (human-readable, companion-ready)
    - course_metadata.json (distances, pars, elevations)
       │
       ▼
[7] COMPANION APP (local web server)
    - Serves build guide on localhost
    - Side-by-side reference: map + instructions
    - Hole-by-hole navigation
    - Checkboxes for completed steps
       │
       ▼
OUTPUT: Builder sits at Xbox with companion on second screen/tablet
        Works hole by hole using build guide
        Manual finishing: vegetation, atmosphere, course conditions, fine sculpting
```

### Where Automation Happens
- Boundary fetch, LiDAR download, DTM generation, heightmap export
- OSM feature extraction, confidence scoring, polygon simplification
- Scale translation, step generation, build pack assembly
- Companion app serving

### Where Human Finishing Is Unavoidable
- All in-game Course Designer work (Xbox controller)
- Vegetation placement and density
- Course conditions (turf colour, firmness, rough length)
- Atmosphere / sky / lighting
- Any feature with <60% confidence score (manual tracing required)
- Fine terrain sculpting in detail areas (bunker lips, green undulation sub-1m)
- Course branding / signage / clubhouse placement

---

## SECTION 3 — SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────┐
│                   COURSE REPLICATOR 2K                  │
│                   (local Python system)                 │
└─────────────────────────────────────────────────────────┘

config.py              ← single config file, all tunable params

pipeline/
├── boundary.py        ← OSM Overpass → course boundary polygon
├── lidar.py           ← INLP/EU-DEM/SRTM fetch → point cloud / raster
├── terrain.py         ← DTM → heightmap → slope map → region classification
├── features.py        ← OSM golf tags → typed feature polygons + scoring
├── translation.py     ← real-world → 2K scale, build instructions
└── qa.py              ← fidelity scoring, distance/elevation validation

companion/
├── app.py             ← Flask app (localhost:5000)
├── templates/
│   ├── base.html
│   ├── index.html     ← course overview
│   └── hole.html      ← per-hole reference + instructions
└── static/
    └── style.css

scripts/
├── run_pipeline.py    ← main CLI entry point
├── fetch_osm.py       ← standalone OSM fetcher
├── fetch_lidar.py     ← standalone LiDAR fetcher
└── export_buildpack.py← standalone build pack exporter

output/
└── {course_slug}/
    ├── heightmap.png
    ├── feature_map.png
    ├── slope_map.png
    ├── holes/
    │   └── hole_XX_overview.png
    ├── build_guide.html
    ├── build_instructions.json
    └── course_metadata.json
```

**Data Flow**:
```
Overpass API ──► boundary.py ──► lidar.py ──► terrain.py ──┐
                     │                                       │
                     └──► features.py ──────────────────────┤
                                                             │
                                                     translation.py
                                                             │
                                                         qa.py
                                                             │
                                                    export_buildpack.py
                                                             │
                                                     companion/app.py
```

**Processing Order**: Sequential. Each stage depends on the previous.

---

## SECTION 4 — DATA SOURCES

### Irish LiDAR Sources

| Source | Resolution | Coverage | Access | Cost |
|--------|-----------|----------|--------|------|
| Irish National LiDAR Programme (INLP) via Tailte Éireann / data.gov.ie | 0.5m | ~70% of Ireland (expanding) | Free, open data | Free |
| EU-DEM v1.1 (Copernicus) | 25m | Full island | Free | Free |
| SRTM v3 (NASA/CGIAR) | 30m | Full island | Free | Free |
| OpenTopography (hosted LIDAR) | Varies | Partial | Free w/ account | Free |

**Strategy**: Try INLP first. Fall back to EU-DEM. Fall back to SRTM. Flag coverage tier in output.

### OSM Golf Feature Usage

Overpass API queries target:
- `landuse=golf_course` — boundary
- `leisure=golf_course` — alternate boundary tag
- `golf=fairway` — fairway polygons
- `golf=green` — putting green polygons
- `golf=bunker` — bunker polygons
- `golf=tee` — tee box polygons
- `golf=water_hazard` — water hazard polygons
- `golf=rough` — rough zone polygons
- `golf=path` — cart paths
- `golf=hole` — hole relations (tee→fairway→green ordering)
- `ref=*` — hole number tags
- `par=*` — par per hole
- `handicap=*` — stroke index

### Aerial Imagery Roles

| Source | Legal Use | Role |
|--------|----------|------|
| Google Maps (browser) | View only | Manual visual reference on second screen |
| Bing Maps (browser) | View only | Manual visual reference on second screen |
| OpenStreetMap tile layers | Open | Background for feature map overlays |
| USGS Earth Explorer | Download OK | Optional high-res imagery for US (not Ireland) |
| Copernicus Sentinel-2 | Free download | 10m optical, used for rough vegetation classification |

**Aerial imagery is not bulk-downloaded.** It is used as a manual visual reference layer in the companion app (embedded map tiles via Leaflet/OSM tiles, which are legally usable).

### Optional Manual References

- Course stroke saver / yardage books (manual entry)
- Club website scorecard (manual entry → JSON)
- Google Street View for tee/green photography reference
- r/golf, golf photography for course character reference
- Club-published course guides / videos

---

## SECTION 5 — GEOSPATIAL PIPELINE

### Boundary Resolution

1. Query Overpass API with course name search → `landuse=golf_course` or `leisure=golf_course`
2. Extract outer boundary as a closed polygon (GeoJSON)
3. Compute bounding box with 200m buffer
4. Store as WGS84 GeoJSON + projected (Irish Transverse Mercator, EPSG:2157)

### LiDAR Acquisition

**INLP Path**:
1. Query Tailte Éireann open data STAC API for tiles intersecting bounding box
2. Download LAZ/LAS point cloud tiles
3. Merge tiles into single LAZ file
4. Filter ground returns (Classification class 2)
5. Thin to 1m spacing if needed

**EU-DEM Fallback**:
1. Identify EU-DEM 5°×5° tiles covering bounding box
2. Download GeoTIFF
3. Clip to bounding box

**SRTM Fallback**:
1. Identify 1°×1° SRTM tiles
2. Download via CGIAR or NASA EarthData
3. Clip to bounding box

### DTM Generation

From point cloud (INLP):
```
pdal pipeline:
  - read LAZ
  - filter ground returns
  - create DTM via triangulation (PDAL writers.gdal)
  - output GeoTIFF at 1m resolution
```

From raster fallback (EU-DEM / SRTM):
```
- clip with gdalwarp to course bounding box
- reproject to EPSG:2157
- resample to consistent resolution
```

### Heightmap Normalisation

```python
# Load DTM
dtm = rasterio.open("dtm.tif").read(1)

# Clip to course boundary (mask non-course cells)
# Get min/max elevation within course boundary
z_min = dtm[mask].min()
z_max = dtm[mask].max()

# Normalise to 0–65535 (16-bit)
heightmap = ((dtm - z_min) / (z_max - z_min) * 65535).astype(np.uint16)

# Export as PNG
Image.fromarray(heightmap).save("heightmap.png")
```

**2K scaling note**: PGA TOUR 2K terrain uses an internal 0–100 height slider. The normalised heightmap maps 0→0 and 65535→100. The builder scales the in-game terrain import reference to match real elevation range.

### Slope and Terrain-Region Extraction

```python
# Compute slope from DTM using numpy gradient
dy, dx = np.gradient(dtm, resolution_m)
slope_deg = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

# Region classification
flat    = slope_deg < 2.0   # green/tee candidates
gentle  = (slope_deg >= 2.0) & (slope_deg < 8.0)  # fairway typical
moderate= (slope_deg >= 8.0) & (slope_deg < 20.0)  # rough/bank
steep   = slope_deg >= 20.0  # hazard lips, dunes, cliffs
```

---

## SECTION 6 — GOLF FEATURE EXTRACTION

### Feature Extraction Pipeline

Each feature type is extracted from OSM, scored for confidence, and stored as GeoJSON.

### Fairways

- Source: `golf=fairway` polygons from OSM
- Processing: Simplify geometry (Douglas-Peucker, 1m tolerance), clip to course boundary
- Confidence scoring:
  - Node count ≥ 8 and area between 500–15000m²: **HIGH (0.9)**
  - Node count 4–7 or unusual area: **MEDIUM (0.6)**
  - Missing or single polygon for entire course: **LOW (0.3)**
- Fallback: Derive from satellite-adjacent slope region (gentle slope + non-bunker)

### Greens

- Source: `golf=green` polygons
- Expected shape: roughly circular/oval, 300–900m² area
- Confidence:
  - Present, correct area, assigned to hole ref: **HIGH (0.95)**
  - Present but no hole ref: **MEDIUM (0.7)**
  - Missing: **LOW (0.0)** — requires manual placement

### Bunkers

- Source: `golf=bunker` polygons
- OSM bunker accuracy is typically ±3–5m positional, shapes approximate
- Confidence:
  - Area 5–500m², multiple bunkers present: **MEDIUM-HIGH (0.75)**
  - Single large polygon or zero: **LOW (0.3)**
- Note: Always flag for manual shape refinement in-game

### Tees

- Source: `golf=tee` polygons
- Multiple tee boxes per hole expected (championship, medal, forward)
- Confidence:
  - Multiple tees with hole ref tags: **HIGH (0.9)**
  - Single tee per hole: **MEDIUM (0.65)**
  - Missing: **LOW (0.0)**

### Water Hazards

- Source: `golf=water_hazard`, `natural=water`, `water=pond/lake/river`
- Confidence:
  - Tagged golf=water_hazard: **HIGH (0.9)**
  - natural=water within course boundary: **MEDIUM (0.7)**
  - Missing, visible in satellite: **LOW (0.1)**

### Paths

- Source: `golf=path`, `highway=path/track` within course boundary
- Used for cart path surface painting and routing reference
- Confidence: MEDIUM across the board; rarely fully mapped in OSM

### Hole Routing

- Source: OSM `golf=hole` relation (tee→fairway→green ordered members)
- Where relations exist: extract ordered hole geometry
- Where missing: compute routing from tee centroid → nearest green centroid → order by distance optimisation
- Output: ordered list of 18 (tee_point, green_point, approximate_length) per hole

### Confidence Scoring Summary

| Feature | Typical OSM Coverage (IE) | Typical Confidence |
|---------|--------------------------|-------------------|
| Boundary | 95% of courses | HIGH |
| Fairways | 80% of courses | MEDIUM-HIGH |
| Greens | 85% of courses | HIGH |
| Bunkers | 70% of courses | MEDIUM |
| Tees | 75% of courses | MEDIUM-HIGH |
| Water | 60% of courses | MEDIUM |
| Paths | 30% of courses | LOW-MEDIUM |
| Hole routing | 50% of courses | MEDIUM |

---

## SECTION 7 — 2K TRANSLATION LAYER

### Real-World → 2K Scale

PGA TOUR 2K's Course Designer operates in a virtual space of approximately:
- Playable terrain: ~1,500 × 1,500 yards (1,372m × 1,372m)
- Maximum course length supported: ~8,000 yards total
- Terrain height range: relative slider 0–100

**Scale mapping**:
```
real_metres_x / course_max_width_metres * 2K_canvas_width_yards = 2K_x_position

# Example: 400m wide course → 437 yards → maps linearly to 2K canvas
# 300m fairway at position 150m from west → 164 yards at position 82 yards in 2K

# Elevation:
real_z_range = z_max - z_min  (e.g. 45m total relief)
2K_height_at_point = (real_z_at_point - z_min) / real_z_range * 100
```

### Hazard Simplification for 2K Tools

2K bunkers are placed as individual ellipse/freeform shapes via controller:
- Complex multi-finger bunkers → represented as 2–4 overlapping ovals
- Exact polygon coordinates → reduced to centre point + approximate radius
- Each bunker gets: centre coords (in 2K grid), approximate width (yards), approximate length (yards), orientation (degrees from north)

### Vegetation Budget

PGA TOUR 2K has object placement limits per course. Budget strictly:
```
Trees:          150–250 per hole (estimate, varies by 2K version)
Bushes/shrubs:  100–200 per hole
Flowers/ground: Paint zones only (no object count impact)

Allocation strategy per hole:
- Championship Irish links: prioritise marram grass zone paint, minimal trees
- Parkland: prioritise tree lines on boundaries
- Heathland: prioritise heather zone paint, sparse trees
```

### Surface and Play Settings

| Feature | 2K Surface Type | Conditions Preset |
|---------|----------------|------------------|
| Fairway polygon | Fairway | Normal |
| Green polygon | Green | Firm (links) / Medium (parkland) |
| Bunker polygon | Bunker | Firm |
| Rough zone | Rough | Thick (links) |
| Water hazard | Water | — |
| Cart path | Path | — |
| Tee polygon | Tee | Normal |
| Fescue/marram grass | Rough (thick) | — |

### Par / Distance / Stroke Index Assignment

Populated from OSM tags where available, manual scorecard entry otherwise:
```json
{
  "hole": 1,
  "par": 4,
  "stroke_index": 7,
  "championship_yards": 410,
  "medal_yards": 395,
  "forward_yards": 345,
  "2k_championship_yards": 410,
  "notes": "Dogleg right at 220y"
}
```

---

## SECTION 8 — ASSISTED BUILD EXECUTION

### Companion App Design

A local Flask web app (localhost:5000) runs on a laptop/tablet placed beside the Xbox.

**Interface per hole**:
```
┌─────────────────────────────────────────────────────────┐
│ HOLE 7 — PAR 4 — 412 YARDS — S.I. 3        [◄ 6] [8 ►] │
├──────────────────────┬──────────────────────────────────┤
│                      │ STEP 1 of 14                [✓]  │
│   [overhead map]     │ TERRAIN: Set terrain to height   │
│   feature_map +      │ preset "Coastal Dunes". Apply    │
│   heightmap overlay  │ base elevation. Raise NE corner  │
│                      │ to ~35 (dune ridge). Flatten     │
│   [slope map]        │ green site to ~22.               │
│                      │──────────────────────────────────│
│                      │ STEP 2 of 14                [✓]  │
│   Confidence:        │ FAIRWAY: Paint fairway starting  │
│   Terrain: HIGH      │ 45y from tee, 35y wide, curving  │
│   Features: MEDIUM   │ left at 180y. Length 280y total. │
│   Bunkers: MEDIUM    │ (see map: blue polygon)          │
│                      │──────────────────────────────────│
│   [Sentinel imagery  │ STEP 3 of 14                [ ]  │
│    link opens in     │ BUNKERS: Place 2 bunkers left    │
│    browser]          │ fairway at 200y. Each ~8y wide.  │
│                      │ 1 bunker right at 230y, ~12y.    │
│                      │ (Confidence: MEDIUM — verify     │
│                      │  against satellite view)         │
└──────────────────────┴──────────────────────────────────┘
```

### Hole-by-Hole Assembly Flow

Recommended order per hole:
1. Set terrain height at key points (tee, landing zone, green, max elevation)
2. Rough-sculpt major landforms (dunes, valleys, hillsides)
3. Paint fairway surface
4. Paint green surface + fine-sculpt undulation
5. Paint tee surface
6. Place bunkers (position, shape, depth)
7. Place water hazards
8. Paint rough zones (thick/thin/fescue)
9. Paint cart path
10. Add tree lines and vegetation zones
11. Fine-sculpt bunker lips, run-offs, chipping areas
12. QA check (walk hole, check distances with in-game rangefinder)

### Step Format

Each build step includes:
- **Category**: TERRAIN / FAIRWAY / GREEN / BUNKER / WATER / ROUGH / VEGETATION / QA
- **Action**: Specific controller action required
- **Measurements**: In 2K yards (translated from real metres)
- **Reference**: Which overlay map element to look at
- **Confidence flag**: HIGH/MEDIUM/LOW with implication
- **Checkbox**: Mark as complete

### Quality-Control Checkpoints

After every 3 holes: walk the holes in preview mode and verify:
- Rangefinder from tee to green matches expected yardage (±5%)
- No surface paint gaps
- Bunker edges blend naturally
- Green is reachable and drains correctly

---

## SECTION 9 — QUALITY ASSURANCE

### Fidelity Scoring (per course, 0–100)

| Dimension | Weight | Measurement |
|-----------|--------|-------------|
| Routing accuracy | 20% | Hole order correct, doglegs in correct direction |
| Distance accuracy | 20% | Each hole within 5% of real yardage |
| Elevation accuracy | 20% | Relative relief within 20% of real range |
| Hazard placement | 15% | Bunkers/water within 10y of correct position |
| Surface accuracy | 10% | Correct surface types applied to correct areas |
| Terrain character | 10% | Dunes/valleys/slopes capture course feel |
| Visual identity | 5% | Vegetation type matches course character |

### Routing Validation
- Check hole count = 18
- Check no hole routing overlaps another hole's fairway
- Check tee-to-green direction vectors match real-world orientation (±15°)
- Check par assignment matches published scorecard

### Distance Validation
```python
for hole in holes:
    tee_to_green_m = geodesic(hole.tee_centroid, hole.green_centroid).meters
    expected_yards = hole.championship_yards
    actual_yards   = tee_to_green_m * 1.094
    error_pct = abs(actual_yards - expected_yards) / expected_yards * 100
    if error_pct > 10:
        flag(hole, f"Distance error {error_pct:.1f}% — check routing")
```

### Elevation Validation
- Compare DTM elevation range (z_max - z_min) against known course profile
- Flag if normalised heightmap range < 5m (likely flat course using SRTM artefact)
- Flag if any green site has slope > 5° (unplayable in 2K)

### Hazard Validation
- Cross-reference bunker count vs scorecard/known data
- Flag any bunker polygon with area > 600m² (likely mis-tagged)
- Flag missing water hazards where course is known to have them

### "Course Feel" Validation (Manual Checklist)
- [ ] Links: Exposed, wind-exposed, dune terrain visible
- [ ] Parkland: Tree-lined fairways, sheltered feel
- [ ] Heathland: Heather rough, open sky, firm ground
- [ ] Clifftop: Elevation drama, sea views
- [ ] River/inland: Water presence, flatter terrain
Builder confirms feel against known photographs/video.

---

## SECTION 10 — MVP BUILD PLAN

### Target Course for MVP
**Old Conna Golf Club** — Well-mapped in OSM, INLP LiDAR coverage exists for County Wicklow, mature parkland character with strong elevation change, 18-hole par 72, publicly known distances.

### Simplest Useful Toolchain

```
Python 3.11+
├── requests          — Overpass API, tile downloads
├── geopandas         — Vector geometry processing
├── shapely           — Polygon operations
├── rasterio          — Raster I/O
├── numpy             — Array operations
├── Pillow            — Heightmap/image export
├── matplotlib        — Map rendering
├── flask             — Companion web app
├── pdal (optional)   — Point cloud processing (only if INLP LiDAR used)
└── pyproj            — Coordinate transformations
```

### File Structure (MVP)

```
Course-Replicator-2K/
├── config.py
├── requirements.txt
├── pipeline/
│   ├── __init__.py
│   ├── boundary.py
│   ├── lidar.py
│   ├── terrain.py
│   ├── features.py
│   ├── translation.py
│   └── qa.py
├── companion/
│   ├── app.py
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   └── hole.html
│   └── static/style.css
├── scripts/
│   ├── run_pipeline.py
│   └── serve_companion.py
├── output/           (git-ignored)
├── DESIGN.md
└── README.md
```

### 30-Day Build Order

| Days | Task |
|------|------|
| 1–3  | Set up repo, install dependencies, verify Overpass API works for Old Conna |
| 4–6  | Build `boundary.py` — fetch and export course boundary polygon |
| 7–9  | Build `features.py` — extract all OSM golf features, confidence scoring |
| 10–13| Build `lidar.py` — INLP tile discovery + download, EU-DEM fallback |
| 14–17| Build `terrain.py` — DTM→heightmap pipeline, slope map |
| 18–20| Build `translation.py` — scale conversion, step generation |
| 21–23| Build `qa.py` — distance validation, fidelity scoring |
| 24–26| Build companion app — Flask, per-hole pages, overlay maps |
| 27–28| Run full pipeline on Old Conna, generate build pack |
| 29–30| Do manual build in 2K using companion app, identify gaps, iterate |

---

## SECTION 11 — HARD TRADEOFFS

### What to Ignore

- Clubhouse buildings (not supported meaningfully in 2K course designer)
- Driving range (outside 18-hole course, not useful)
- Sub-metre bunker lip detail (can't be captured in OSM or expressed accurately via controller)
- Exact tree species (irrelevant — 2K has fixed tree catalogue)
- Crowd/spectator stands (placed manually if wanted)
- Exact green undulation shapes (1-2cm swales; slope data at 1m is insufficient)

### What to Fake

- Vegetation density: Use 2K zone painting with approximate rough/fescue/heather zones based on terrain region classification (steep zones → heather, flat zones → short rough)
- Bunker depth: All bunkers set to "deep" in links, "medium" in parkland — no depth data in OSM or LiDAR
- Green slopes: Infer from DTM slope map — green marked as "tilted left→right" if slope vector points that way; micro-undulation is manual
- Cart paths: Approximate from OSM where missing; straight paths between tees and greens as fallback

### What to Simplify

- Complex multi-finger bunkers → 2–3 overlapping simple shapes
- Island/peninsula green shapes → approximate oval
- Burn/stream routing → simplified polyline
- Very long par 5s on Irish links (>560y) → flagged, builder manually extends terrain

### What Matters Most for Realism (Priority Order)

1. **Routing direction and dogleg character** — the feel of each hole's strategy
2. **Yardage per hole** — nothing kills realism faster than wrong distances
3. **Relative elevation** — especially for dune/clifftop courses
4. **Bunker positions** — not shapes, but strategic placement (left/right, greenside/fairway)
5. **Green character** — size, tilt direction, approach angle
6. **Rough/surface zoning** — links fescue vs parkland grass feel
7. **Vegetation type** — trees vs open vs heather (zone painted)
8. Exact bunker shapes (low priority — controller precision limits this anyway)
9. Individual tree placement (lowest priority — time sink)

---

## SECTION 12 — FINAL RECOMMENDATION

### Best Overall System Design

A **local Python pipeline + local Flask companion app**. No cloud. No SaaS. No CI/CD. One person runs it on a laptop, generates a build pack, serves the companion app on a tablet or second monitor, and builds on Xbox.

The pipeline is **data-in, build-pack-out**. The companion app is **build-pack-in, builder-guided-out**.

### Best Tool Choices

| Task | Tool | Reason |
|------|------|--------|
| OSM data | Overpass API (requests) | Free, no key, full golf tagging |
| Terrain (primary) | Tailte Éireann INLP | Best resolution for Ireland |
| Terrain (fallback) | EU-DEM via Copernicus | Free, 25m, full coverage |
| Terrain processing | rasterio + numpy | Lightweight, no GDAL install hell |
| Vector processing | geopandas + shapely | Mature, well-documented |
| Map rendering | matplotlib | Simple, no JS required |
| Companion UI | Flask + Leaflet.js | Minimal, local, no framework complexity |
| Point cloud (if INLP) | PDAL | Industry standard, handles LAZ |
| Output format | PNG + JSON + HTML | Universal, no special software needed |

### Best Project Scope for One Person

- Support 10–15 Irish courses total (prioritise those with good OSM + INLP coverage)
- Process one course at a time
- One pipeline run = one build pack
- No parallelism needed
- Total codebase target: ~2,000 lines of Python + ~500 lines of HTML/CSS

### Most Likely Path to a Result That Actually Works

1. **Prove the data first**: Manually download OSM data for Old Conna, check quality. Manually download a LiDAR tile, confirm coverage. If either is bad, re-scope.
2. **Build boundary.py and features.py first**: These are pure API calls and geometry — lowest risk, immediate validation.
3. **Build terrain.py with EU-DEM fallback first**: Don't wait for INLP pipeline to work. EU-DEM at 25m is enough to validate the heightmap workflow.
4. **Build the companion app early (day 18–20)**: Getting the output usable beside Xbox quickly reveals what data is actually needed.
5. **Build one course in 2K before finishing the tool**: The act of building Old Conna with a half-finished tool will reveal what information is missing and what's irrelevant.
6. **Iterate on the pipeline from real build experience**: The builder knows what they needed and didn't have. That drives the backfill roadmap.

### Honest Final Assessment

This system will save 60–75% of the manual research and guesswork involved in replicating an Irish golf course in PGA TOUR 2K. It will not eliminate the in-game build work — that's a platform constraint, not a tooling failure. A determined single builder with this system can produce a recognisable, playable, correctly-routed Irish golf course in 2K in roughly 8–15 hours of in-game work per course (versus 25–40 hours without it). The terrain and routing automation alone justify the build effort. This is the right scope. Do not expand it.

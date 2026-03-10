"""
config.py — Central configuration for Course Replicator 2K.

All tunable parameters live here. Edit this file before running the pipeline.

Directory layout (auto-created at import time):
    output/
    assets/
    assets/routing_diagrams/
    assets/yardage_books/
    cache/
    temp/
"""

from pathlib import Path

# ─── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
OUTPUT_DIR  = str(BASE_DIR / "output")
ASSETS_DIR  = str(BASE_DIR / "assets")
DIAGRAM_DIR = str(BASE_DIR / "assets" / "routing_diagrams")
YARDAGE_DIR = str(BASE_DIR / "assets" / "yardage_books")
CACHE_DIR   = str(BASE_DIR / "cache")
TEMP_DIR    = str(BASE_DIR / "temp")

# ─── Auto-create required directories ────────────────────────────────────────
for _d in [OUTPUT_DIR, ASSETS_DIR, DIAGRAM_DIR, YARDAGE_DIR, CACHE_DIR, TEMP_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)

# ─── Overpass API ────────────────────────────────────────────────────────────
OVERPASS_URL      = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT  = 120
OSM_TIMEOUT       = 60

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
OVERPASS_SERVERS = OVERPASS_ENDPOINTS   # alias

# ─── Nominatim geocoder ───────────────────────────────────────────────────────
NOMINATIM_URL     = "https://nominatim.openstreetmap.org/search"
NOMINATIM_TIMEOUT = 12

# ─── Course name aliases ──────────────────────────────────────────────────────
COURSE_ALIASES = {
    "old conna golf club":   "Old Conna Golf Course",
    "old conna":             "Old Conna Golf Course",
    "old conna golf course": "Old Conna Golf Course",
    "the k club":            "The K Club",
    "k club":                "The K Club",
    "mount juliet":          "Mount Juliet Golf Club",
    "mount juliet golf":     "Mount Juliet Golf Club",
}

# ─── Terrain / DEM sources (priority order) ───────────────────────────────────
#
# DEM_SOURCES controls which elevation sources are tried and in what order.
# Each entry is a string key matching a source handler in lidar.py.
# Remove or comment out sources you don't want attempted.
#
DEM_SOURCES = [
    "inlp",           # 1. Irish National LiDAR — 0.5m (Ireland only)
    "opentopography", # 2. OpenTopography global LiDAR/DEM — 1-30m
    "copernicus",     # 3. Copernicus GLO-30 — 30m global
    "srtm",           # 4. NASA SRTM — 30m global
    "mapzen",         # 5. Mapzen Terrarium — ~38m (final fallback)
]

# Irish National LiDAR Programme (Tailte Éireann)
INLP_STAC_URL   = "https://data.gov.ie/api/3/action/package_search"
INLP_WCS_URL    = "https://wms.tailte.ie/inspire/ows"
INLP_ARCGIS_URL = (
    "https://gis.epa.ie/arcgis/rest/services/EPA/LiDAR_DTM_IE/ImageServer/exportImage"
)

# OpenTopography public API — free key from https://opentopography.org
OPENTOPO_API_URL = "https://portal.opentopography.org/API/globaldem"
OPENTOPO_API_KEY = ""    # set your key here for higher rate limits

# Copernicus GLO-30 (AWS Open Data Registry)
COPERNICUS_DEM_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com"
)
# Alternate URL (sometimes faster for Europe):
COPERNICUS_DEM_URL_ALT = (
    "https://opentopography.s3.sdsc.edu/raster/CopernicusDEM/CopernicusDEM30"
)

# EU-DEM v1.1 Copernicus (25m, Europe only, older dataset)
EUDEM_BASE_URL = "https://opentopography.s3.sdsc.edu/raster/EU_DEM/EU_DEM_be_5deg"

# NASA SRTM 30m via OpenTopography
SRTM_API_URL = "https://portal.opentopography.org/API/globaldem"
SRTM_URL     = SRTM_API_URL    # alias
SRTM_API_KEY = ""              # same key as OPENTOPO_API_KEY

# Mapzen Terrarium tiles — global, no key required
MAPZEN_TILE_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)

# ─── Terrain quality validation thresholds ────────────────────────────────────
DEM_MIN_VALID_PCT  = 10.0   # % of pixels that must be non-nodata before accepting
DEM_POST_CLIP_PCT  = 5.0    # lenient threshold after clipping to course bbox
DEM_MIN_VARIANCE   = 0.1    # m² — reject completely flat DEMs (likely corrupt)
DEM_MAX_SLOPE_DEG  = 85.0   # reject DEMs with unrealistic spike slopes

# ─── Satellite imagery sources (priority order) ───────────────────────────────
#
# SATELLITE_SOURCES controls which tile providers are tried.
# Order matters — first working provider wins.
#
SATELLITE_SOURCES = [
    "google",    # 1. Google Satellite
    "esri",      # 2. ESRI World Imagery
    "bing",      # 3. Bing Aerial
]

# Google Satellite (unofficial XYZ tiles)
GOOGLE_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

# ESRI World Imagery (official, no key for read-only)
ESRI_TILE_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)

# Bing Maps aerial
BING_TILE_URL = (
    "https://t.ssl.ak.dynamic.tiles.virtualearth.net/comp/ch/{quadkey}"
    "?mkt=en-IE&it=A&shading=hill&og=2177&n=z"
)

# Mosaic output size
SATELLITE_MOSAIC_SIZE = 2048   # pixels square

# ─── Terrain processing ───────────────────────────────────────────────────────
DTM_RESOLUTION_M   = 1.0
BOUNDARY_BUFFER_M  = 300

SLOPE_FLAT_MAX     = 2.0
SLOPE_GENTLE_MAX   = 8.0
SLOPE_MODERATE_MAX = 20.0

# ─── 2K course designer parameters ───────────────────────────────────────────
TK2_CANVAS_YARDS    = 1372
TK2_MAX_TOTAL_YARDS = 8000
TK2_HEIGHT_MIN      = 0
TK2_HEIGHT_MAX      = 100
MIN_ELEVATION_RELIEF_M = 2.0

# ─── Course type presets ──────────────────────────────────────────────────────
COURSE_TYPE_PRESETS = {
    "links": {
        "rough": "Thick Fescue", "green_speed": "Fast", "ground": "Firm",
        "fairway_width": 40, "rough_width": 20, "tree_density": 0.05,
    },
    "parkland": {
        "rough": "Thick Rough", "green_speed": "Medium", "ground": "Soft",
        "fairway_width": 45, "rough_width": 25, "tree_density": 0.65,
    },
    "heathland": {
        "rough": "Heather", "green_speed": "Medium-Fast", "ground": "Medium",
        "fairway_width": 38, "rough_width": 18, "tree_density": 0.30,
    },
    "clifftop": {
        "rough": "Thick Fescue", "green_speed": "Fast", "ground": "Firm",
        "fairway_width": 35, "rough_width": 15, "tree_density": 0.10,
    },
    "inland": {
        "rough": "Medium Rough", "green_speed": "Medium", "ground": "Soft",
        "fairway_width": 42, "rough_width": 22, "tree_density": 0.45,
    },
    "desert": {
        "rough": "Desert Sand", "green_speed": "Fast", "ground": "Very Firm",
        "fairway_width": 32, "rough_width": 10, "tree_density": 0.05,
    },
    "tropical": {
        "rough": "Tropical Rough", "green_speed": "Medium", "ground": "Soft",
        "fairway_width": 44, "rough_width": 20, "tree_density": 0.80,
    },
}

# ─── Feature extraction ───────────────────────────────────────────────────────
CONFIDENCE_HIGH   = 0.80
CONFIDENCE_MEDIUM = 0.55
CONFIDENCE_LOW    = 0.30

FEATURE_AREA_RANGES = {
    "green":   (250,   900),
    "fairway": (500,  20000),
    "bunker":  (  5,    600),
    "tee":     ( 50,    600),
    "water":   ( 50, 500000),
}

SIMPLIFY_TOLERANCE_M = 1.0

# ─── Vision detection area thresholds (real-world m²) ────────────────────────
MIN_BUNKER_AREA_M2  =   40
MAX_BUNKER_AREA_M2  =  600
MIN_GREEN_AREA_M2   =  200
MAX_GREEN_AREA_M2   = 2500
MIN_FAIRWAY_AREA_M2 = 1000
MIN_WATER_AREA_M2   =   50
MAX_WATER_AREA_M2   = 500000

# Short aliases
MIN_GREEN_AREA   = MIN_GREEN_AREA_M2
MIN_BUNKER_AREA  = MIN_BUNKER_AREA_M2
MIN_FAIRWAY_AREA = MIN_FAIRWAY_AREA_M2
MIN_WATER_AREA   = MIN_WATER_AREA_M2

PGA_2K_CANVAS_M = 1255

# ─── Tee detection ────────────────────────────────────────────────────────────
TEE_SEARCH_MIN_M = 5
TEE_SEARCH_MAX_M = 80

# ─── Feature fusion priority ──────────────────────────────────────────────────
# Controls merge order in feature_fusion.py
# Lower index = higher priority
FUSION_PRIORITY = [
    "routing_diagram",   # 1. routing diagram AI (highest fidelity)
    "yardage_book",      # 2. yardage book AI
    "satellite_vision",  # 3. satellite vision detection
    "osm",               # 4. OpenStreetMap (baseline)
]

# ─── Yardage book AI ──────────────────────────────────────────────────────────
YARDAGE_BOOK_ENABLED = True
YARDAGE_BOOK_MIN_CONFIDENCE = 0.35

# ─── OCR ─────────────────────────────────────────────────────────────────────
# Path to tesseract executable.
#   Windows : r"C:\Program Files\Tesseract-OCR\tesseract.exe"  (set below)
#   Linux   : "" → auto-detect via PATH (tesseract installed via apt/brew)
#   macOS   : "" → auto-detect (installed via `brew install tesseract`)
# If TESSERACT_CMD is set and the binary does not exist at that path, the
# pipeline will log a warning and skip OCR rather than crash.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─── Routing diagram ─────────────────────────────────────────────────────────
ROUTING_DIAGRAM_ENABLED = True

# ─── Optional pipeline stages ─────────────────────────────────────────────────
ENABLE_TEE_DETECTION = True
ENABLE_ML_VISION     = False
ENABLE_SAM           = False
SAM_CHECKPOINT       = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE       = "vit_b"

# ─── Vegetation budgets ───────────────────────────────────────────────────────
VEGETATION_BUDGET = {
    "trees": 200, "bushes": 150, "rocks": 50, "misc": 100,
}

# ─── Cache validation ─────────────────────────────────────────────────────────
INVALID_CACHE_MIN_BYTES = 10

# ─── Output ───────────────────────────────────────────────────────────────────
HEIGHTMAP_SIZE_PX = 1024
FEATURE_MAP_DPI   = 150

# ─── Web servers ─────────────────────────────────────────────────────────────
COMPANION_HOST  = "localhost"
COMPANION_PORT  = 5000
DIGITIZER_HOST  = "0.0.0.0"
DIGITIZER_PORT  = 5050

# ─── Coordinate reference systems ────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"
CRS_ITM   = "EPSG:2157"

# ─── Known courses ────────────────────────────────────────────────────────────
KNOWN_COURSES = {
    "old conna golf club":   [-6.138, 53.191, -6.125, 53.198],
    "old conna golf course": [-6.138, 53.191, -6.125, 53.198],
    "old conna":             [-6.138, 53.191, -6.125, 53.198],
    "powerscourt golf club": [-6.207, 53.162, -6.160, 53.192],
    "druids glen golf club": [-6.101, 53.070, -6.057, 53.097],
    "druids heath golf club":[-6.108, 53.062, -6.063, 53.090],
    "laytown and bettystown":[-6.250, 53.694, -6.215, 53.718],
    "the k club":            [-6.670, 53.305, -6.625, 53.330],
    "mount juliet golf club":[-7.220, 52.555, -7.175, 52.580],
    "adare manor golf club": [-8.820, 52.562, -8.775, 52.588],
    "fota island golf club": [-8.315, 51.893, -8.265, 51.918],
    "carton house golf club":[-6.588, 53.373, -6.543, 53.398],
}

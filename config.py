"""
config.py — Central configuration for Course Replicator 2K.

All tunable parameters live here. Edit this file before running the pipeline.

Directory layout (created automatically at import time):
    output/
    assets/
    assets/routing_diagrams/
    cache/
    temp/
"""

from pathlib import Path

# ─── Base paths ───────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()   # project root
OUTPUT_DIR = str(BASE_DIR / "output")
ASSETS_DIR = str(BASE_DIR / "assets")
DIAGRAM_DIR = str(BASE_DIR / "assets" / "routing_diagrams")
CACHE_DIR  = str(BASE_DIR / "cache")
TEMP_DIR   = str(BASE_DIR / "temp")

# ─── Auto-create required directories ────────────────────────────────────────
for _d in [OUTPUT_DIR, ASSETS_DIR, DIAGRAM_DIR, CACHE_DIR, TEMP_DIR]:
    Path(_d).mkdir(parents=True, exist_ok=True)

# ─── Overpass API ────────────────────────────────────────────────────────────
# Primary endpoint retained for backward compatibility.
# boundary.py rotates across OVERPASS_ENDPOINTS per request, falling back on
# the next endpoint whenever a 429 or connection error is received.
OVERPASS_URL     = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 120  # seconds
OSM_TIMEOUT      = 60   # seconds (lighter queries via overpy)

# Endpoint rotation pool (shuffled randomly per request by boundary.py)
OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
# Alias used by older/external modules
OVERPASS_SERVERS = OVERPASS_ENDPOINTS

# ─── Nominatim geocoder ───────────────────────────────────────────────────────
# Used as the primary geocoder in boundary.py.
# The public instance has a 1 req/s rate limit; always include a User-Agent.
NOMINATIM_URL     = "https://nominatim.openstreetmap.org/search"
NOMINATIM_TIMEOUT = 12   # seconds

# ─── Course name aliases ──────────────────────────────────────────────────────
# Maps common/colloquial input names → canonical name used for geocoding.
# Keys must be lowercase; values are the name sent to Nominatim/Overpass.
COURSE_ALIASES = {
    "old conna golf club":   "Old Conna Golf Course",
    "old conna":             "Old Conna Golf Course",
    "old conna golf course": "Old Conna Golf Course",
    "the k club":            "The K Club",
    "k club":                "The K Club",
    "mount juliet":          "Mount Juliet Golf Club",
    "mount juliet golf":     "Mount Juliet Golf Club",
}

# ─── Terrain / elevation sources ─────────────────────────────────────────────
# Irish National LiDAR Programme (Tailte Éireann) STAC endpoint
INLP_STAC_URL = "https://data.gov.ie/api/3/action/package_search"
INLP_WCS_URL  = "https://wms.tailte.ie/inspire/ows"  # fallback WCS endpoint

# EU-DEM Copernicus (25m coverage, full Ireland)
EUDEM_BASE_URL = "https://opentopography.s3.sdsc.edu/raster/EU_DEM/EU_DEM_be_5deg"

# SRTM 30m via OpenTopography API
SRTM_API_URL   = "https://portal.opentopography.org/API/globaldem"
SRTM_URL       = SRTM_API_URL   # alias used by newer modules
SRTM_API_KEY   = ""  # Free key from opentopography.org — optional for SRTM

# Mapzen Terrarium tiles — global 1-arc-second elevation, no key required
MAPZEN_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# ─── Satellite tile sources ───────────────────────────────────────────────────
# Bing aerial imagery (replace {key} with a valid Bing Maps API key if needed)
BING_TILE_URL   = "https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1"
# Google aerial imagery (unofficial, may require API key in production)
GOOGLE_TILE_URL = "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}"

# ─── Terrain processing ───────────────────────────────────────────────────────
# Resolution to resample DTM to (metres)
DTM_RESOLUTION_M = 1.0

# Buffer around course boundary for terrain fetch (metres)
BOUNDARY_BUFFER_M = 300

# Slope thresholds (degrees)
SLOPE_FLAT_MAX     = 2.0
SLOPE_GENTLE_MAX   = 8.0
SLOPE_MODERATE_MAX = 20.0
# >= SLOPE_MODERATE_MAX is classified as steep

# ─── 2K course designer parameters ───────────────────────────────────────────
# PGA TOUR 2K playable canvas (yards). Approximate — varies by 2K edition.
TK2_CANVAS_YARDS = 1372
TK2_MAX_TOTAL_YARDS = 8000
TK2_HEIGHT_MIN = 0
TK2_HEIGHT_MAX = 100

# Minimum elevation relief to use (metres). Below this, terrain is treated as flat.
MIN_ELEVATION_RELIEF_M = 2.0

# ─── Course type presets ──────────────────────────────────────────────────────
# Each preset defines:
#   rough           — 2K rough texture name
#   green_speed     — putting green speed descriptor
#   ground          — firmness descriptor
#   fairway_width   — typical fairway width in metres
#   rough_width     — typical rough width in metres
#   tree_density    — relative tree density (0.0–1.0)
COURSE_TYPE_PRESETS = {
    "links": {
        "rough": "Thick Fescue",
        "green_speed": "Fast",
        "ground": "Firm",
        "fairway_width": 40,
        "rough_width": 20,
        "tree_density": 0.05,
    },
    "parkland": {
        "rough": "Thick Rough",
        "green_speed": "Medium",
        "ground": "Soft",
        "fairway_width": 45,
        "rough_width": 25,
        "tree_density": 0.65,
    },
    "heathland": {
        "rough": "Heather",
        "green_speed": "Medium-Fast",
        "ground": "Medium",
        "fairway_width": 38,
        "rough_width": 18,
        "tree_density": 0.30,
    },
    "clifftop": {
        "rough": "Thick Fescue",
        "green_speed": "Fast",
        "ground": "Firm",
        "fairway_width": 35,
        "rough_width": 15,
        "tree_density": 0.10,
    },
    "inland": {
        "rough": "Medium Rough",
        "green_speed": "Medium",
        "ground": "Soft",
        "fairway_width": 42,
        "rough_width": 22,
        "tree_density": 0.45,
    },
    "desert": {
        "rough": "Desert Sand",
        "green_speed": "Fast",
        "ground": "Very Firm",
        "fairway_width": 32,
        "rough_width": 10,
        "tree_density": 0.05,
    },
    "tropical": {
        "rough": "Tropical Rough",
        "green_speed": "Medium",
        "ground": "Soft",
        "fairway_width": 44,
        "rough_width": 20,
        "tree_density": 0.80,
    },
}

# ─── Feature extraction ───────────────────────────────────────────────────────
# Confidence thresholds
CONFIDENCE_HIGH   = 0.80
CONFIDENCE_MEDIUM = 0.55
CONFIDENCE_LOW    = 0.30

# Expected area ranges per feature type (m²)
FEATURE_AREA_RANGES = {
    "green":   (250, 900),
    "fairway": (500, 20000),
    "bunker":  (5, 600),
    "tee":     (50, 600),
    "water":   (50, 500000),
}

# Polygon simplification tolerance (metres)
SIMPLIFY_TOLERANCE_M = 1.0

# ─── Vision detection area thresholds (real-world m²) ────────────────────────
MIN_BUNKER_AREA_M2  = 40
MAX_BUNKER_AREA_M2  = 600
MIN_GREEN_AREA_M2   = 200
MAX_GREEN_AREA_M2   = 2500
MIN_FAIRWAY_AREA_M2 = 1000
MIN_WATER_AREA_M2   = 50
MAX_WATER_AREA_M2   = 500000

# Short aliases for modules that use the unqualified names
MIN_GREEN_AREA   = MIN_GREEN_AREA_M2
MIN_BUNKER_AREA  = MIN_BUNKER_AREA_M2
MIN_FAIRWAY_AREA = MIN_FAIRWAY_AREA_M2
MIN_WATER_AREA   = MIN_WATER_AREA_M2

# PGA 2K canvas in real-world metres (≈ 1372 yards × 0.9144)
PGA_2K_CANVAS_M = 1255

# ─── Tee detection ────────────────────────────────────────────────────────────
TEE_SEARCH_MIN_M = 5
TEE_SEARCH_MAX_M = 80

# ─── OCR / pytesseract ────────────────────────────────────────────────────────
# Path to the tesseract executable. Leave empty to let pytesseract auto-detect
# (works on Linux/Mac). On Windows set to e.g. r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD = ""

# ─── Routing diagram AI ───────────────────────────────────────────────────────
ROUTING_DIAGRAM_ENABLED = True

# ─── Optional pipeline stages ─────────────────────────────────────────────────
ENABLE_TEE_DETECTION = True
ENABLE_ML_VISION = False
ENABLE_SAM     = False
SAM_CHECKPOINT = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# ─── Vegetation budgets (2K object limits) ────────────────────────────────────
VEGETATION_BUDGET = {
    "trees":   200,
    "bushes":  150,
    "rocks":   50,
    "misc":    100,
}

# ─── Cache validation ─────────────────────────────────────────────────────────
INVALID_CACHE_MIN_BYTES = 10

# ─── Output ───────────────────────────────────────────────────────────────────
HEIGHTMAP_SIZE_PX = 1024
FEATURE_MAP_DPI   = 150

# ─── Companion / digitizer web servers ───────────────────────────────────────
COMPANION_HOST  = "localhost"
COMPANION_PORT  = 5000
DIGITIZER_HOST  = "0.0.0.0"
DIGITIZER_PORT  = 5050   # avoids clash with companion on 5000

# ─── Coordinate reference systems ────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"
CRS_ITM    = "EPSG:2157"   # Irish Transverse Mercator

# ─── Known courses ────────────────────────────────────────────────────────────
KNOWN_COURSES = {
    "old conna golf club":      [-6.138, 53.191, -6.125, 53.198],
    "old conna golf course":    [-6.138, 53.191, -6.125, 53.198],
    "old conna":                [-6.138, 53.191, -6.125, 53.198],
    "powerscourt golf club":    [-6.207, 53.162, -6.160, 53.192],
    "druids glen golf club":    [-6.101, 53.070, -6.057, 53.097],
    "druids heath golf club":   [-6.108, 53.062, -6.063, 53.090],
    "laytown and bettystown":   [-6.250, 53.694, -6.215, 53.718],
    "the k club":               [-6.670, 53.305, -6.625, 53.330],
    "mount juliet golf club":   [-7.220, 52.555, -7.175, 52.580],
    "adare manor golf club":    [-8.820, 52.562, -8.775, 52.588],
    "fota island golf club":    [-8.315, 51.893, -8.265, 51.918],
    "carton house golf club":   [-6.588, 53.373, -6.543, 53.398],
}

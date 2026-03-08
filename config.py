"""
config.py — Central configuration for Course Replicator 2K.
All tunable parameters live here. Edit this file before running the pipeline.
"""

# ─── Overpass API ────────────────────────────────────────────────────────────
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_TIMEOUT = 120  # seconds

# ─── LiDAR sources ────────────────────────────────────────────────────────────
# Irish National LiDAR Programme (Tailte Éireann) STAC endpoint
INLP_STAC_URL = "https://data.gov.ie/api/3/action/package_search"
INLP_WCS_URL  = "https://wms.tailte.ie/inspire/ows"  # fallback WCS endpoint

# EU-DEM Copernicus (25m coverage, full Ireland)
EUDEM_BASE_URL = "https://opentopography.s3.sdsc.edu/raster/EU_DEM/EU_DEM_be_5deg"

# SRTM 30m via OpenTopography API
SRTM_API_URL   = "https://portal.opentopography.org/API/globaldem"
SRTM_API_KEY   = ""  # Free key from opentopography.org — optional for SRTM

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
# 2K25: ~1372 × 1372 yards. Set to a safe value that works for full 18-hole layout.
TK2_CANVAS_YARDS = 1372

# Maximum course length supported (yards total yardage, all 18 holes)
TK2_MAX_TOTAL_YARDS = 8000

# Height slider range (0–100 internal units)
TK2_HEIGHT_MIN = 0
TK2_HEIGHT_MAX = 100

# Minimum elevation relief to use (metres). Below this, terrain is treated as flat.
MIN_ELEVATION_RELIEF_M = 2.0

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

# ─── Vegetation budgets (2K object limits, approximate) ─────────────────────
VEGETATION_BUDGET = {
    "trees":   200,    # per hole, max
    "bushes":  150,
    "rocks":   50,
    "misc":    100,
}

# Course type presets (used in translation layer)
COURSE_TYPE_PRESETS = {
    "links":      {"rough": "Thick Fescue", "green_speed": "Fast", "ground": "Firm"},
    "parkland":   {"rough": "Thick Rough",  "green_speed": "Medium", "ground": "Soft"},
    "heathland":  {"rough": "Heather",      "green_speed": "Medium-Fast", "ground": "Medium"},
    "clifftop":   {"rough": "Thick Fescue", "green_speed": "Fast", "ground": "Firm"},
    "inland":     {"rough": "Medium Rough", "green_speed": "Medium", "ground": "Soft"},
}

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "output"

# Heightmap image size (pixels). 1024×1024 is sufficient for reference use.
HEIGHTMAP_SIZE_PX = 1024

# Feature map DPI
FEATURE_MAP_DPI = 150

# Companion web app
COMPANION_HOST = "localhost"
COMPANION_PORT = 5000

# ─── Coordinate reference systems ─────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"
CRS_ITM    = "EPSG:2157"   # Irish Transverse Mercator — used for all metric calculations

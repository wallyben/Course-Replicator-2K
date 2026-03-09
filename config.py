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

# ─── Vision detection area thresholds (real-world m²) ────────────────────────
# Contours outside these ranges are discarded as false positives after projecting
# from pixel space into metric coordinates.
MIN_BUNKER_AREA_M2  = 40      # m² — minimum credible bunker
MAX_BUNKER_AREA_M2  = 600     # m² — maximum credible bunker
MIN_GREEN_AREA_M2   = 200     # m² — minimum putting green
MAX_GREEN_AREA_M2   = 2500    # m² — maximum putting green
MIN_FAIRWAY_AREA_M2 = 1000    # m² — minimum fairway strip

# PGA 2K canvas in real-world metres (≈ 1372 yards × 0.9144)
PGA_2K_CANVAS_M = 1255

# ─── Tee detection ─────────────────────────────────────────────────────────────
# Distance range from fairway start to search for tee boxes (metres)
TEE_SEARCH_MIN_M = 5
TEE_SEARCH_MAX_M = 80

# ─── Optional ML vision refinement ────────────────────────────────────────────
# Set True to enable ML-based segmentation refinement (pipeline/ml_vision.py).
# Requires optional heavy dependencies (torch, torchvision, etc.).
# When False the pipeline behaves exactly as without this flag.
ENABLE_ML_VISION = False

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

# ─── Known courses ────────────────────────────────────────────────────────────
# Hardcoded bounding boxes [min_lon, min_lat, max_lon, max_lat] for courses
# that are missing or mis-tagged in OSM. These are used as an automatic fallback
# when OSM resolution fails, so --bbox is not needed for these courses.
# Add more as you discover OSM coverage gaps.
KNOWN_COURSES = {
    # Key: lowercase normalised name → [min_lon, min_lat, max_lon, max_lat]
    "old conna golf club":      [-6.155, 53.182, -6.110, 53.208],
    "old conna":                [-6.155, 53.182, -6.110, 53.208],
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

"""Configuration for the candidate‑example queue system.
All values are simple constants for the prototype; they can later be loaded from
YAML if needed.
"""

# Maximum number of pending items to retain; older entries are pruned.
MAX_PENDING_ITEMS = 1000

# Retention policy for approved datasets – keep the last N versions.
MAX_DATASET_VERSIONS = 5

# Thresholds for the quality filter (example values).
MIN_CONFIDENCE = 0.6          # Minimum average hand‑tracking confidence.
MIN_WINDOW_LENGTH = 10        # Minimum number of frames in the landmark sequence.
REQUIRE_BOTH_HANDS = True

# Regression tolerance: if the regression set error increases by more than this
# fraction, the new model must not be promoted.
REGRESSION_TOLERANCE = 0.05

# Directory layout (relative to project root).
BASE_DATA_DIR = "data"
PENDING_DIR = f"{BASE_DATA_DIR}/queue/pending"
VERSIONS_DIR = f"{BASE_DATA_DIR}/versions"

# Model registry file.
REGISTRY_FILE = f"{BASE_DATA_DIR}/model_registry.json"

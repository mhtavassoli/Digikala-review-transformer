"""
Global configuration for Project 3 (Digikala Review Transformer).
All constants that affect reproducibility live here.
"""
from pathlib import Path

# -------------------------------------------------------------------
# Reproducibility constants (MUST NOT CHANGE after splits are built)
# -------------------------------------------------------------------
SPLIT_SEED = 2026
MASK_SEED  = "digikala-semi-supervised-v1"
MASK_RATE  = 0.20

# -------------------------------------------------------------------
# Rating-band thresholds (Task B, Section 6)
# Bands are inclusive on the low side, exclusive on the high side,
# EXCEPT the last band which is inclusive on both ends (0..5 scale).
# -------------------------------------------------------------------
RATING_BANDS = [
    # (low, high, label, ordinal_index)
    (0.0, 2.5, "low",    0),
    (2.5, 4.0, "medium", 1),
    (4.0, 5.0, "high",   2),
]

# -------------------------------------------------------------------
# Canonical recommendation classes (Task A, Section 5)
# The KEY of the map is the string as it appears in the raw CSV.
# The VALUE is the canonical label used everywhere else.
# NOTE: The actual unique values must be confirmed in the data audit.
#       This map is the DEFAULT; extend it after inspecting the file.
# -------------------------------------------------------------------
RECOMMENDATION_CANONICAL = {
    "recommended":     "recommended",
    "not_recommended": "not_recommended",
    "no_idea":         "no_idea",
}
RECOMMENDATION_LABELS = ["recommended", "not_recommended", "no_idea"]
RATING_BAND_LABELS    = ["low", "medium", "high"]

# -------------------------------------------------------------------
# Modeling
# -------------------------------------------------------------------
PRETRAINED_ENCODER = "HooshvareLab/bert-fa-base-uncased"
MAX_SEQ_LEN        = 256          # justified by A4 length analysis
BATCH_SIZE         = 32
GRAD_ACCUM         = 2
EPOCHS             = 4
LR_ENCODER         = 2e-5         # smaller LR for pretrained weights
LR_HEAD            = 1e-4         # larger LR for task heads
WEIGHT_DECAY       = 0.01
WARMUP_RATIO       = 0.06
PSEUDO_CONF_THRESH = 0.90
PSEUDO_EPOCHS      = 2

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR    = ROOT / "data"
ARTIFACTS   = ROOT / "artifacts"
REPORTS     = ROOT / "reports"
SPLITS_DIR  = ROOT / "splits"

for d in (DATA_DIR, ARTIFACTS, REPORTS, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)
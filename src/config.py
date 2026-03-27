from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "papers"
REPORTS_DIR = BASE_DIR / "reports"
BUCKET_FILE = BASE_DIR / "bucket.json"

EARLY_EXIT_THRESHOLD = 4

COMPOSITE_WEIGHTS = {
    "statistical": 0.30,
    "methodology": 0.30,
    "ml": 0.20,
    "microstructure": 0.20,
}

CRITICAL_FLAGS = {
    "lookahead_bias",
    "leaky_features",
    "p_hacking",
    "data_snooping_bias",
}

CRITIC_TIMEOUT_SECONDS = 120

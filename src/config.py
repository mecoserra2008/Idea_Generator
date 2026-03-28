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

# --- Discovery defaults ---
DISCOVERY_DEFAULT_QUERIES = [
    "quantitative trading machine learning",
    "algorithmic trading deep learning",
    "statistical arbitrage",
    "portfolio optimization reinforcement learning",
    "market microstructure neural network",
    "high frequency trading",
]

DISCOVERY_DEFAULT_CATEGORIES = [
    "q-fin.TR",   # Trading and Market Microstructure
    "q-fin.PM",   # Portfolio Management
    "q-fin.ST",   # Statistical Finance
    "q-fin.CP",   # Computational Finance
    "cs.LG",      # Machine Learning (cross-listed)
    "stat.ML",    # Machine Learning (cross-listed)
]

DISCOVERY_MAX_RESULTS = 50        # per query
DISCOVERY_DAYS_BACK = 7           # default lookback window
DISCOVERY_RATE_DELAY = 3.0        # seconds between API calls (arXiv asks for 3s)

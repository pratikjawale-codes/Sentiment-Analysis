from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DEFAULT_DATASET = DATA_DIR / "sample_tweets.csv"
LEGACY_DATASET = ROOT / "Tweets.csv"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"

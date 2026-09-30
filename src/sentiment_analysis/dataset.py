from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DEFAULT_DATASET, LEGACY_DATASET


def load_training_data(dataset_path: str | Path | None = None) -> pd.DataFrame:
    path = Path(dataset_path) if dataset_path else (LEGACY_DATASET if LEGACY_DATASET.exists() else DEFAULT_DATASET)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    text_col = "text" if "text" in df.columns else next((c for c in df.columns if c.lower() == "tweet"), None)
    label_col = "airline_sentiment" if "airline_sentiment" in df.columns else "label"

    if text_col is None or label_col not in df.columns:
        raise ValueError(f"Dataset must include a text column and a label column. Found: {list(df.columns)}")

    result = df[[text_col, label_col]].dropna().copy()
    result = result.rename(columns={text_col: "text", label_col: "label"})
    result["label"] = result["label"].astype(str).str.strip().str.lower()
    return result

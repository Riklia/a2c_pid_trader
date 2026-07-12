"""Filesystem cache for computed feature tables."""
import hashlib
import json
from pathlib import Path

import pandas as pd


DEFAULT_CACHE_DIR = Path(".cache/features")


def _cache_key(params: dict) -> str:
    """Deterministic hash of the parameter dict, used as a filename."""
    encoded = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_path_for(params: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Returns the parquet path a feature table with these params would live at."""
    return cache_dir / f"features_{_cache_key(params)}.parquet"


def load_from_cache(params: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame | None:
    """Returns the cached feature table for these params, or None if not cached."""
    path = cache_path_for(params, cache_dir)
    if not path.exists():
        return None
    return pd.read_parquet(path)


def save_to_cache(df: pd.DataFrame, params: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Writes the feature table to the cache, returns the file path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path_for(params, cache_dir)
    df.to_parquet(path)
    return path

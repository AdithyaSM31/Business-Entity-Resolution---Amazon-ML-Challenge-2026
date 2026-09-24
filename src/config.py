"""Shared paths and settings for every pipeline stage.

Owner: Adithya. Everyone imports from here, so change it only through a PR.
To point at a different folder on your machine, set the environment variable instead of editing this file.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("BER_DATA_DIR", ROOT / "dataset"))  # holds train/ and test/
CACHE_DIR = Path(os.environ.get("BER_CACHE_DIR", ROOT / "cache"))  # intermediate files, see docs/INTERFACES.md
MODEL_DIR = Path(os.environ.get("BER_MODEL_DIR", ROOT / "models"))
OUTPUT_DIR = Path(os.environ.get("BER_OUTPUT_DIR", ROOT / "output"))

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)

SEED = 42
N_FOLDS = 5

# Fraction of Source 1 entities kept for quick development on laptops without a GPU.
# Anything submitted or logged in docs/experiments.csv must use 1.0.
SAMPLE_FRAC = float(os.environ.get("BER_SAMPLE_FRAC", "1.0"))
N_JOBS = int(os.environ.get("BER_N_JOBS", str(os.cpu_count() or 4)))


def cache_path(name: str) -> Path:
    """Path of a cache file (creating its folder). File names are fixed in docs/INTERFACES.md."""
    path = CACHE_DIR / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

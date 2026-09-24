"""Reading the challenge TSVs and writing the two submission files.

Owner: Arushi (writers). The readers are shared: never read the raw TSVs any other way,
because pandas' defaults silently turn empty fields and names like "NA" into NaN.
"""
import csv
import zlib
from pathlib import Path

import pandas as pd

from . import config

GT_S1_COL = "source1_entity_id"
GT_MATCH_COL = "matched_entity_ids"


def _read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                     quoting=csv.QUOTE_NONE, encoding="utf-8")
    with open(path, encoding="utf-8") as f:
        n_data_lines = sum(1 for line in f if line.strip()) - 1
    if len(df) != n_data_lines:
        raise ValueError(f"{path}: parsed {len(df)} rows but the file has {n_data_lines} data lines")
    return df


def read_source(split: str, source: int) -> pd.DataFrame:
    """One source file: entity_id, business_name, business_address, country (all str, "" when missing)."""
    return _read_tsv(config.DATA_DIR / split / f"{split}_source{source}.tsv")


def read_all_records() -> pd.DataFrame:
    """All six source files stacked, with extra columns `split` ("train"/"test") and `source` (1/2/3).

    Entity IDs may repeat between train and test, so always key records by (split, entity_id).
    """
    frames = []
    for split in config.SPLITS:
        for source in config.SOURCES:
            df = read_source(split, source)
            df.insert(0, "source", source)
            df.insert(0, "split", split)
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def read_ground_truth() -> pd.DataFrame:
    """Train labels, one row per Source 1 entity: s1_id and matches (list of str, empty for singletons)."""
    gt = _read_tsv(config.DATA_DIR / "train" / "train_ground_truth.tsv")
    return pd.DataFrame({
        "s1_id": gt[GT_S1_COL],
        "matches": gt[GT_MATCH_COL].map(lambda s: [x.strip() for x in s.split(",") if x.strip()]),
    })


def in_dev_sample(s1_id: str) -> bool:
    """Deterministic subsample of Source 1 entities, identical on every laptop (see config.SAMPLE_FRAC)."""
    return zlib.crc32(s1_id.encode()) % 10_000 < config.SAMPLE_FRAC * 10_000


def write_id_lists(s1_ids: list[str], id_lists: dict[str, list[str]], list_col: str, path: Path) -> None:
    """Write matching_results.tsv (list_col="matched_entity_ids") or candidate_pairs.tsv
    (list_col="candidate_entity_ids").

    The validator rejects the file unless:
    - header is `source1_entity_id<TAB><list_col>` and there is exactly one row per test S1 entity
      (every country, France included), in the order of `s1_ids`
    - IDs are comma-joined with no spaces or quotes, no duplicates, S2-/S3- IDs from the test set only
    - an entity with no IDs gets an empty string (never "nan"); UTF-8 with "\\n" line endings
    """
    raise NotImplementedError("AR-2")

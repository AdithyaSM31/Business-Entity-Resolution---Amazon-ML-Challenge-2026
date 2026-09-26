"""Labels and validation folds over training Source 1 entities. Owner: Arushi (AR-1).

Outputs:
  cache/labels.parquet  s1_id, cand_id, label (=1 for every ground-truth pair)
  cache/folds.parquet   s1_id, country, n_true, fold (0..N_FOLDS-1), stratified by country x match-count bucket
S2/S3 records are never split: every fold is scored against the full S2/S3 pool, like the test set.
Leave-one-country-out (LOCO) validation uses the `country` column directly.
"""


"""Labels and validation folds over training Source 1 entities. Owner: Arushi (AR-1)."""

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config, io_utils


def make_labels_and_folds() -> None:
    """Build train labels and S1-level stratified/LOCO folds."""
    truth_df = io_utils.read_ground_truth()
    train_s1 = io_utils.read_source("train", 1)

    # Train work is sampled through the shared deterministic sampler; S2/S3 are never sampled.
    if config.SAMPLE_FRAC < 1.0:
        keep = train_s1["entity_id"].map(io_utils.in_dev_sample)
        train_s1 = train_s1.loc[keep].copy()
        truth_df = truth_df[truth_df["s1_id"].isin(train_s1["entity_id"])].copy()

    truth_df = truth_df.drop_duplicates("s1_id", keep="first")
    truth_by_s1 = dict(zip(truth_df["s1_id"], truth_df["matches"]))

    # Include every retained S1, including singletons whose match list is empty.
    folds_df = train_s1[["entity_id", "country"]].rename(columns={"entity_id": "s1_id"}).copy()
    folds_df["n_true"] = folds_df["s1_id"].map(lambda s1_id: len(truth_by_s1.get(s1_id, []))).astype("int32")
    folds_df["stratum"] = folds_df["country"] + "_" + folds_df["n_true"].clip(upper=3).astype(str)

    splitter = StratifiedKFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.SEED)
    folds_df["fold"] = -1
    for fold, (_, valid_idx) in enumerate(splitter.split(folds_df, folds_df["stratum"])):
        folds_df.iloc[valid_idx, folds_df.columns.get_loc("fold")] = fold

    # Deterministic integer country codes for leave-one-country-out validation.
    countries = sorted(folds_df["country"].unique())
    country_codes = {country: code for code, country in enumerate(countries)}
    folds_df["loco_fold"] = folds_df["country"].map(country_codes).astype("int16")
    folds_df = folds_df[["s1_id", "country", "n_true", "fold", "loco_fold"]]

    labels = truth_df[["s1_id", "matches"]].explode("matches", ignore_index=True)
    labels = labels.rename(columns={"matches": "cand_id"})
    labels = labels[labels["cand_id"].notna()][["s1_id", "cand_id"]].drop_duplicates()
    labels["label"] = 1
    labels = labels[["s1_id", "cand_id", "label"]]

    config.cache_path("labels.parquet").parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(config.cache_path("labels.parquet"), index=False)
    folds_df.to_parquet(config.cache_path("folds.parquet"), index=False)

    print(f"labels: {len(labels):,} rows")
    print(f"folds: {len(folds_df):,} S1 entities")
    for fold, group in folds_df.groupby("fold", sort=True):
        singleton_rate = float((group["n_true"] == 0).mean())
        print(f"fold {fold}: {len(group):,} S1, singleton_rate={singleton_rate:.4f}")


if __name__ == "__main__":
    make_labels_and_folds()

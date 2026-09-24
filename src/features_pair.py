"""Pairwise similarity features for every candidate pair. Owner: Bhanu (BH-4, BH-6).

Input:  cache/records.parquet, cache/candidates_{split}.parquet
Output: cache/feat_pair_{split}.parquet  (s1_id, cand_id, fn_* name features, fa_* address features)
Rows must be exactly the candidate rows; join on (s1_id, cand_id), never on row order.
Parallelize with multiprocessing; on Windows the entry point needs the `if __name__ == "__main__":` guard.
"""
import argparse

import pandas as pd


def build_pair_features(split: str) -> pd.DataFrame:
    raise NotImplementedError("BH-4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    build_pair_features(parser.parse_args().split)

"""Context and competition features. Owner: Siva (SI-4, SI-5).

fc_*  (SI-4, from blocking only): name/address frequency (chains, malls), near-duplicate count per S1,
      rank within the S1's candidates, reverse rank (rank of this S1 among the S1s that list this record),
      gap to the best candidate, pass flags and pass scores.
      Output: cache/feat_ctx_{split}.parquet
fc2_* (SI-5, from stage-1 predictions; on train these MUST be the out-of-fold predictions):
      rank and margin by probability, mutual best, best competing probability, cross-source support
      (max over the S1's other candidates a of p(s1, a) * sim(a, cand)).
      Output: cache/feat_ctx2_{split}_{run_id}.parquet
"""
import argparse

import pandas as pd


def build_context_features(split: str) -> pd.DataFrame:
    raise NotImplementedError("SI-4")


def build_stage2_features(split: str, run_id: str) -> pd.DataFrame:
    raise NotImplementedError("SI-5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--stage1-run", help="run_id of stage-1 predictions; builds fc2_* features when given")
    args = parser.parse_args()
    if args.stage1_run:
        build_stage2_features(args.split, args.stage1_run)
    else:
        build_context_features(args.split)

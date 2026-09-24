"""LightGBM matching model, stage 1 and stage 2. Owner: Adithya (AD-3, AD-7, AD-8).

Input:  candidates + every cache/feat_*_{split}.parquet joined on (s1_id, cand_id), labels, folds
Output: models/{run_id}/fold{k}.txt, models/{run_id}/features.json, models/{run_id}/importance.csv
        cache/preds/{run_id}_train.parquet  (out-of-fold probabilities for every train candidate)
Country is never a model feature (France is unseen in train); it is only a blocking partition.
run_id format: YYYYMMDD-HHMM_short-description, and every run gets a row in docs/experiments.csv.
"""
import argparse


def train(run_id: str, stage: int, seeds: list[int]) -> None:
    raise NotImplementedError("AD-3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    train(args.run_id, args.stage, args.seeds)

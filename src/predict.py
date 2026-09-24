"""Test-set inference: average the fold models of a run. Owner: Adithya (AD-3).

Output: cache/preds/{run_id}_test.parquet  (s1_id, cand_id, prob), one row per test candidate.
"""
import argparse


def predict(run_id: str) -> None:
    raise NotImplementedError("AD-3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    predict(parser.parse_args().run_id)

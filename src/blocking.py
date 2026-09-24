"""Candidate generation (blocking). Owner: Siva (SI-1, SI-3, SI-6).

Input:  cache/records.parquet, plus cache/emb/dense_neighbors_{split}.parquet from Adithya for the dense pass
Output: cache/candidates_{split}.parquet, the FINAL pruned candidate set (schema: docs/INTERFACES.md).
That file is exactly what the model scores and exactly what goes into candidate_pairs.tsv.

Each pass runs within every country value found in the data (an open set: never list countries by hand),
separately for S1->S2 and S1->S3. Target: >= 99% pair recall at roughly 30-50 candidates per S1.
"""
import argparse

import pandas as pd

PASSES = ("tfidf_name", "tfidf_full", "rare_token", "addr_key", "dense", "reverse")


def run_pass(name: str, split: str) -> pd.DataFrame:
    """One blocking pass -> DataFrame[s1_id, cand_id, score]."""
    raise NotImplementedError("SI-1 / SI-3")


def generate_candidates(split: str) -> pd.DataFrame:
    """Union of all passes, pruned to the top-N per S1, saved to cache/candidates_{split}.parquet."""
    raise NotImplementedError("SI-3")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    generate_candidates(parser.parse_args().split)

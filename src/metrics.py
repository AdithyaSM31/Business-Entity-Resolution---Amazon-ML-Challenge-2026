"""The leaderboard metric and blocking diagnostics. Owner: Arushi (AR-1).

Tests to write first:
  PDF example: pred {S2-00047, S2-00193, S3-00812}, true {S2-00047, S3-00812} -> 0.714
  both empty -> 1.0;  pred empty, true non-empty -> 0.0;  pred non-empty, true empty -> 0.0
"""
import pandas as pd


def f05_entity(pred: set[str], true: set[str]) -> float:
    """F0.5 for one S1 entity: 1.25 * P * R / (0.25 * P + R), with the empty-set rules above."""
    raise NotImplementedError("AR-1")


def macro_f05(pred: dict[str, list[str]], true: dict[str, list[str]]) -> float:
    """Mean of f05_entity over every S1 in `true`; an S1 missing from `pred` counts as an empty prediction."""
    raise NotImplementedError("AR-1")


def blocking_report(candidates: pd.DataFrame, truth: dict[str, list[str]]) -> dict:
    """pair_recall, oracle_f05 (a perfect classifier on these candidates), mean/p95 candidates per S1."""
    raise NotImplementedError("AR-1")

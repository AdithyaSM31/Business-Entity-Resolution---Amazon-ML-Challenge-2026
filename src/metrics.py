"""The leaderboard metric and blocking diagnostics. Owner: Arushi (AR-1).

Tests to write first:
  PDF example: pred {S2-00047, S2-00193, S3-00812}, true {S2-00047, S3-00812} -> 0.714
  both empty -> 1.0;  pred empty, true non-empty -> 0.0;  pred non-empty, true empty -> 0.0
"""
import pandas as pd
import numpy as np
import pandas as pd


def f05_entity(pred: set[str], true: set[str]) -> float:
    """Return entity-level F0.5 using the challenge's empty-set rules."""
    pred = set(pred)
    true = set(true)
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0

    tp = len(pred & true)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(true)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_f05(
    pred: dict[str, list[str]],
    true: dict[str, list[str]],
    return_per_entity: bool = False,
):
    """Mean entity-level F0.5 over every S1 key in ``true``."""
    scores = pd.Series(
        {s1_id: f05_entity(set(pred.get(s1_id, [])), set(matches))
         for s1_id, matches in true.items()},
        dtype=float,
    )
    score = float(scores.mean()) if not scores.empty else 0.0
    return (score, scores) if return_per_entity else score


def blocking_report(candidates: pd.DataFrame, truth: dict[str, list[str]]) -> dict:
    """Report candidate recall, entity ceiling, oracle F0.5 and candidate counts."""
    required = {"s1_id", "cand_id"}
    missing = required - set(candidates.columns)
    if missing:
        raise ValueError(f"candidates is missing columns: {sorted(missing)}")

    candidate_sets = (
        candidates.groupby("s1_id", sort=False)["cand_id"]
        .agg(lambda values: set(values))
        .to_dict()
    )

    true_sets = {s1_id: set(matches) for s1_id, matches in truth.items()}
    total_true_pairs = sum(len(matches) for matches in true_sets.values())
    found_pairs = sum(len(matches & candidate_sets.get(s1_id, set()))
                      for s1_id, matches in true_sets.items())
    pair_recall = found_pairs / total_true_pairs if total_true_pairs else 1.0

    entity_ceiling = (
        sum(matches <= candidate_sets.get(s1_id, set()) for s1_id, matches in true_sets.items())
        / len(true_sets)
        if true_sets else 0.0
    )

    oracle_pred = {
        s1_id: list(matches & candidate_sets.get(s1_id, set()))
        for s1_id, matches in true_sets.items()
    }
    oracle_f05 = float(macro_f05(oracle_pred, {k: list(v) for k, v in true_sets.items()}))

    counts = pd.Series(0, index=pd.Index(true_sets.keys(), dtype=object), dtype=float)
    if candidate_sets:
        observed_counts = pd.Series({s1_id: len(ids) for s1_id, ids in candidate_sets.items()}, dtype=float)
        counts = counts.add(observed_counts, fill_value=0.0)

    return {
        "pair_recall": float(pair_recall),
        "entity_ceiling": float(entity_ceiling),
        "oracle_f05": oracle_f05,
        "mean_candidates": float(counts.mean()) if not counts.empty else 0.0,
        "p50_candidates": float(np.percentile(counts, 50)) if not counts.empty else 0.0,
        "p95_candidates": float(np.percentile(counts, 95)) if not counts.empty else 0.0,
    }

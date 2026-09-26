"""Evaluate out-of-fold entity-resolution predictions. Owner: Arushi (AR-3).

CLI:
    python -m src.evaluate --run-id <run_id> [--params params.json]

Train predictions must be out-of-fold and stored as
cache/preds/<run_id>_train.parquet with columns s1_id, cand_id, prob and
optionally fold. Ground truth and cache/folds.parquet provide the evaluation
universe and groups.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from . import config, io_utils, metrics, postprocess


LOG_COLUMNS = [
    "run_id", "date_ist", "owner", "change_tested", "pair_recall",
    "oracle_f05", "cv_f05", "loco_us_to_india", "loco_india_to_us",
    "public_lb", "submitted", "notes",
]


def _truth() -> dict[str, list[str]]:
    gt = io_utils.read_ground_truth()
    return dict(zip(gt["s1_id"].astype(str), gt["matches"]))


def _load_params(path: str | None) -> dict:
    if path is None:
        return {"threshold": 0.5}
    with open(path, encoding="utf-8") as f:
        params = json.load(f)
    if not isinstance(params, dict):
        raise ValueError("params JSON must contain an object")
    return params


def _threshold_select(preds: pd.DataFrame, threshold: float) -> dict[str, list[str]]:
    selected = preds.loc[preds["prob"] >= float(threshold)]
    return selected.groupby("s1_id")["cand_id"].apply(list).to_dict()


def _select(preds: pd.DataFrame, params: dict, using_default: bool) -> dict[str, list[str]]:
    try:
        selected = postprocess.select_matches(preds, params)
    except NotImplementedError:
        if not using_default:
            raise
        selected = _threshold_select(preds, float(params["threshold"]))
    if not isinstance(selected, dict):
        raise TypeError("postprocess.select_matches must return dict[str, list[str]]")
    return {str(k): list(v) for k, v in selected.items()}


def _subset_truth(truth: dict[str, list[str]], ids) -> dict[str, list[str]]:
    return {s1_id: truth.get(s1_id, []) for s1_id in ids}


def _subset_pred(pred: dict[str, list[str]], ids) -> dict[str, list[str]]:
    return {s1_id: pred.get(s1_id, []) for s1_id in ids}


def _score(pred: dict[str, list[str]], truth: dict[str, list[str]], ids) -> float:
    return float(metrics.macro_f05(_subset_pred(pred, ids), _subset_truth(truth, ids)))


def _bucket(n_true: int) -> str:
    if n_true <= 0:
        return "0"
    if n_true == 1:
        return "1"
    if n_true == 2:
        return "2"
    return "3+"


def _print_group_scores(name: str, groups: list[tuple[str, list[str]]], pred, truth) -> None:
    print(f"\n{name}")
    print("group\tn\tmacro_f05")
    for group_name, ids in groups:
        if not ids:
            continue
        print(f"{group_name}\t{len(ids):,}\t{_score(pred, truth, ids):.6f}")


def _source_scores(preds: pd.DataFrame, selected: dict[str, list[str]], truth: dict[str, list[str]], ids) -> None:
    print("\nPer source")
    print("source\tn\tmacro_f05")
    for source in (2, 3):
        prefix = f"S{source}-"
        source_pred = {
            s1_id: [x for x in selected.get(s1_id, []) if str(x).startswith(prefix)]
            for s1_id in ids
        }
        source_truth = {
            s1_id: [x for x in truth.get(s1_id, []) if str(x).startswith(prefix)]
            for s1_id in ids
        }
        print(f"S{source}\t{len(ids):,}\t{float(metrics.macro_f05(source_pred, source_truth)):.6f}")


def _pair_recall(pred: dict[str, list[str]], truth: dict[str, list[str]], ids) -> float:
    true_pairs = sum(len(truth.get(i, [])) for i in ids)
    hit_pairs = sum(len(set(pred.get(i, [])) & set(truth.get(i, []))) for i in ids)
    return hit_pairs / true_pairs if true_pairs else 1.0


def _oracle_f05(pred: dict[str, list[str]], truth: dict[str, list[str]], ids) -> float:
    # For AR-3 logging, the selected set is treated as the available candidate set.
    oracle = {i: [x for x in truth.get(i, []) if x in set(pred.get(i, []))] for i in ids}
    return _score(oracle, truth, ids)


def _empty_rates(pred: dict[str, list[str]], truth: dict[str, list[str]], ids) -> tuple[float, float]:
    singleton_ids = [i for i in ids if len(truth.get(i, [])) == 1]
    non_singleton_ids = [i for i in ids if len(truth.get(i, [])) > 1]
    singleton_accuracy = (
        sum(not pred.get(i, []) for i in singleton_ids) / len(singleton_ids)
        if singleton_ids else float("nan")
    )
    non_singleton_empty = (
        sum(not pred.get(i, []) for i in non_singleton_ids) / len(non_singleton_ids)
        if non_singleton_ids else float("nan")
    )
    return singleton_accuracy, non_singleton_empty


def _evaluate_train(run_id: str, params: dict, using_default: bool):
    pred_path = config.cache_path(f"preds/{run_id}_train.parquet")
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing OOF predictions: {pred_path}")
    folds_path = config.cache_path("folds.parquet")
    if not folds_path.exists():
        raise FileNotFoundError(f"Missing folds: {folds_path}; run `python -m src.folds` first")

    preds = pd.read_parquet(pred_path)
    required = {"s1_id", "cand_id", "prob"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"{pred_path}: missing columns {sorted(missing)}")
    preds["s1_id"] = preds["s1_id"].astype(str)
    preds["cand_id"] = preds["cand_id"].astype(str)
    preds["prob"] = pd.to_numeric(preds["prob"], errors="raise")

    folds = pd.read_parquet(folds_path)
    required_folds = {"s1_id", "country", "n_true", "fold"}
    missing = required_folds - set(folds.columns)
    if missing:
        raise ValueError(f"{folds_path}: missing columns {sorted(missing)}")
    folds["s1_id"] = folds["s1_id"].astype(str)
    folds["n_true"] = pd.to_numeric(folds["n_true"], errors="raise").astype(int)
    folds["fold"] = pd.to_numeric(folds["fold"], errors="raise").astype(int)

    truth = _truth()
    ids = folds["s1_id"].tolist()
    selected = _select(preds, params, using_default)
    cv_f05 = _score(selected, truth, ids)

    print(f"run_id: {run_id}")
    print(f"OOF predictions: {len(preds):,} rows")
    print(f"evaluation entities: {len(ids):,}")
    print(f"selection params: {json.dumps(params, sort_keys=True)}")
    print(f"\nOverall macro F0.5: {cv_f05:.6f}")

    fold_groups = [(str(k), folds.loc[folds["fold"] == k, "s1_id"].tolist()) for k in sorted(folds["fold"].unique())]
    country_groups = [(str(k), g["s1_id"].tolist()) for k, g in folds.groupby("country", sort=True)]
    bucket_groups = [(b, folds.loc[folds["n_true"].map(_bucket) == b, "s1_id"].tolist()) for b in ("0", "1", "2", "3+")]
    _print_group_scores("Per fold", fold_groups, selected, truth)
    _print_group_scores("Per country", country_groups, selected, truth)
    _print_group_scores("Per n_true bucket", bucket_groups, selected, truth)
    _source_scores(preds, selected, truth, ids)

    singleton_accuracy, non_singleton_empty = _empty_rates(selected, truth, ids)
    print(f"\nSingleton accuracy (true singleton predicted empty): {singleton_accuracy:.6f}")
    print(f"Non-singleton empty rate: {non_singleton_empty:.6f}")

    return preds, folds, truth, selected, cv_f05


def _evaluate_loco(run_id: str, params: dict, using_default: bool, folds: pd.DataFrame, truth: dict):
    path = config.cache_path(f"preds/{run_id}_loco_train.parquet")
    if not path.exists():
        print(f"\nLOCO: no file found at {path}; skipping.")
        return {}
    preds = pd.read_parquet(path)
    required = {"s1_id", "cand_id", "prob"}
    missing = required - set(preds.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    preds["s1_id"] = preds["s1_id"].astype(str)
    preds["cand_id"] = preds["cand_id"].astype(str)
    preds["prob"] = pd.to_numeric(preds["prob"], errors="raise")
    selected = _select(preds, params, using_default)

    print("\nLeave-one-country-out macro F0.5")
    print("held_out_country\tn\tmacro_f05")
    scores = {}
    for country, group in folds.groupby("country", sort=True):
        ids = group["s1_id"].tolist()
        score = _score(selected, truth, ids)
        scores[str(country)] = score
        print(f"{country}\t{len(ids):,}\t{score:.6f}")
    return scores


def _append_experiment(run_id: str, cv_f05: float, loco_scores: dict, pair_recall: float, oracle_f05: float, params: dict) -> None:
    path = Path(__file__).resolve().parents[1] / "docs" / "experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        columns = list(rows[0].keys()) if rows else LOG_COLUMNS
    else:
        rows, columns = [], LOG_COLUMNS
    for col in LOG_COLUMNS:
        if col not in columns:
            columns.append(col)
    row = {col: "" for col in columns}
    row.update({
        "run_id": run_id,
        "date_ist": datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d"),
        "owner": "Arushi",
        "change_tested": "AR-3 evaluation",
        "pair_recall": f"{pair_recall:.6f}",
        "oracle_f05": f"{oracle_f05:.6f}",
        "cv_f05": f"{cv_f05:.6f}",
        "public_lb": "",
        "submitted": "",
        "notes": json.dumps({"params": params, "loco": loco_scores}, sort_keys=True),
    })
    for country, score in loco_scores.items():
        if country.lower() in {"us", "usa", "united states", "united states of america"}:
            row["loco_us_to_india"] = f"{score:.6f}"
        if country.lower() in {"india", "in"}:
            row["loco_india_to_us"] = f"{score:.6f}"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(row)
    print(f"\nExperiment logged: {path}")


def evaluate(run_id: str, params_path: str | None = None) -> None:
    params = _load_params(params_path)
    using_default = params_path is None
    preds, folds, truth, selected, cv_f05 = _evaluate_train(run_id, params, using_default)
    ids = folds["s1_id"].tolist()
    loco_scores = _evaluate_loco(run_id, params, using_default, folds, truth)
    pair_recall = _pair_recall(selected, truth, ids)
    oracle_f05 = _oracle_f05(selected, truth, ids)
    _append_experiment(run_id, cv_f05, loco_scores, pair_recall, oracle_f05, params)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate out-of-fold entity-resolution predictions")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--params", default=None, help="JSON file containing postprocess parameters")
    args = parser.parse_args()
    evaluate(args.run_id, args.params)


if __name__ == "__main__":
    main()

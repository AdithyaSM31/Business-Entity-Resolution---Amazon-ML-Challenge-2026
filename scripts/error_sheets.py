"""AR-6: build false-positive / false-negative error sheets.

Usage:
    python scripts/error_sheets.py --run-id <run_id>

Inputs:
    cache/preds/<run_id>_train.parquet
    cache/candidates_train.parquet
    cache/records.parquet
    models/<run_id>/postprocess.json

Outputs:
    docs/errors/<run_id>_fp.csv
    docs/errors/<run_id>_fn.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    from src import io_utils
    from src.postprocess import select_matches, fit_isotonic_oof
except ImportError:
    # Allows running as: python scripts/error_sheets.py ... from repo root.
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src import io_utils
    from src.postprocess import select_matches, fit_isotonic_oof


REQUIRED_PRED = {"s1_id", "cand_id", "prob"}
OUTPUT_COLUMNS = [
    "s1_id", "cand_id", "prob", "rank_within_s1", "country", "source",
    "s1_raw_name", "s1_raw_address", "cand_raw_name", "cand_raw_address",
    "s1_name_core", "s1_addr_norm", "cand_name_core", "cand_addr_norm",
    "true_match_count",
]
FN_COLUMNS = OUTPUT_COLUMNS[:-1] + ["true_match_count", "miss_type"]


def _read_params(root: Path, run_id: str) -> dict:
    paths = [
        root / "models" / run_id / "postprocess.json",
        root / "models" / "postprocess.json",
    ]
    for path in paths:
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(
        "Could not find postprocess params. Tried: "
        + ", ".join(str(p) for p in paths)
    )


def _read_predictions(root: Path, run_id: str) -> pd.DataFrame:
    path = root / "cache" / "preds" / f"{run_id}_train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"OOF predictions not found: {path}")
    p = pd.read_parquet(path)
    missing = REQUIRED_PRED - set(p.columns)
    if missing:
        raise ValueError(f"OOF predictions missing columns: {sorted(missing)}")
    p = p.copy()
    p["s1_id"] = p["s1_id"].astype(str)
    p["cand_id"] = p["cand_id"].astype(str)
    p["prob"] = pd.to_numeric(p["prob"], errors="coerce")
    p = p.dropna(subset=["prob"])
    p = p.drop_duplicates(["s1_id", "cand_id"], keep="last")
    p["rank_within_s1"] = p.groupby("s1_id")["prob"].rank(
        method="min", ascending=False
    ).astype(int)
    return p


def _read_candidates(root: Path) -> pd.DataFrame:
    path = root / "cache" / "candidates_train.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Candidates not found: {path}")
    c = pd.read_parquet(path)
    required = {"s1_id", "cand_id"}
    missing = required - set(c.columns)
    if missing:
        raise ValueError(f"Candidates missing columns: {sorted(missing)}")
    return c[["s1_id", "cand_id"]].astype(str).drop_duplicates()


def _read_records(root: Path) -> pd.DataFrame:
    path = root / "cache" / "records.parquet"
    if path.exists():
        r = pd.read_parquet(path)
    else:
        # Raw TSVs are read through the project's io_utils contract.
        r = io_utils.read_all_records()

    required = {"split", "entity_id", "business_name", "business_address", "country",
                "name_core", "addr_norm"}
    missing = required - set(r.columns)
    if missing:
        raise ValueError(f"records missing columns: {sorted(missing)}")
    return r


def _truth(root: Path) -> dict[str, list[str]]:
    # Ground truth is read through the project's reader as required by the playbook.
    gt = io_utils.read_ground_truth()
    if isinstance(gt, dict):
        return {str(k): [str(x) for x in v] for k, v in gt.items()}

    if not isinstance(gt, pd.DataFrame):
        raise TypeError("io_utils.read_ground_truth() must return a dict or DataFrame")

    cols = set(gt.columns)
    if {"source1_entity_id", "matched_entity_ids"}.issubset(cols):
        return {
            str(row.source1_entity_id): _split_ids(row.matched_entity_ids)
            for row in gt.itertuples(index=False)
        }
    if {"s1_id", "cand_id"}.issubset(cols):
        out: dict[str, list[str]] = {}
        for row in gt.itertuples(index=False):
            out.setdefault(str(row.s1_id), []).append(str(row.cand_id))
        return out
    raise ValueError(f"Unsupported ground-truth columns: {sorted(cols)}")


def _split_ids(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _prepare_predictions_for_selection(
    preds: pd.DataFrame,
    params: dict,
    truth: dict[str, list[str]],
) -> pd.DataFrame:
    """Recreate AR-5 cross-fitted calibrated probabilities when needed."""
    method = str(params.get("method", "threshold")).lower()
    if method not in {"expected_f", "expected_f_gate"}:
        return preds

    if "fold" not in preds.columns:
        raise ValueError("AR-5 error sheets require 'fold' in OOF predictions")

    cal = fit_isotonic_oof(preds, truth)
    cal["fold"] = preds["fold"].to_numpy()
    return cal


def _selected_pairs(preds: pd.DataFrame, params: dict, truth: dict[str, list[str]]) -> set[tuple[str, str]]:
    work = _prepare_predictions_for_selection(preds, params, truth)
    selected = select_matches(work, params)
    return {(str(s1), str(cand)) for s1, ids in selected.items() for cand in ids}


def _record_lookup(records: pd.DataFrame) -> dict[tuple[str, str], dict]:
    out = {}
    for row in records.itertuples(index=False):
        key = (str(row.split), str(row.entity_id))
        out[key] = {
            "business_name": getattr(row, "business_name", ""),
            "business_address": getattr(row, "business_address", ""),
            "country": getattr(row, "country", ""),
            "name_core": getattr(row, "name_core", ""),
            "addr_norm": getattr(row, "addr_norm", ""),
        }
    return out


def _source(cand_id: str) -> str:
    if str(cand_id).startswith("S2-"):
        return "S2"
    if str(cand_id).startswith("S3-"):
        return "S3"
    return ""


def _enrich(rows: pd.DataFrame, records: pd.DataFrame, truth: dict[str, list[str]]) -> pd.DataFrame:
    lookup = _record_lookup(records)
    data = []

    for row in rows.itertuples(index=False):
        s1 = str(row.s1_id)
        cand = str(row.cand_id)
        s1r = lookup.get(("train", s1), {})
        cr = lookup.get(("train", cand), {})
        # Some implementations may store records with an empty/other split label.
        if not s1r:
            s1r = next((v for (sp, eid), v in lookup.items() if eid == s1), {})
        if not cr:
            cr = next((v for (sp, eid), v in lookup.items() if eid == cand), {})

        data.append({
            "s1_id": s1,
            "cand_id": cand,
            "prob": float(row.prob) if pd.notna(row.prob) else np.nan,
            "rank_within_s1": getattr(row, "rank_within_s1", np.nan),
            "country": s1r.get("country", cr.get("country", "")),
            "source": _source(cand),
            "s1_raw_name": s1r.get("business_name", ""),
            "s1_raw_address": s1r.get("business_address", ""),
            "cand_raw_name": cr.get("business_name", ""),
            "cand_raw_address": cr.get("business_address", ""),
            "s1_name_core": s1r.get("name_core", ""),
            "s1_addr_norm": s1r.get("addr_norm", ""),
            "cand_name_core": cr.get("name_core", ""),
            "cand_addr_norm": cr.get("addr_norm", ""),
            "true_match_count": len(truth.get(s1, [])),
        })

    return pd.DataFrame(data)


def _write_error_sheets(root: Path, run_id: str, preds: pd.DataFrame, candidates: pd.DataFrame,
                        truth: dict[str, list[str]], params: dict, records: pd.DataFrame) -> None:
    out_dir = root / "docs" / "errors"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_set = set(map(tuple, candidates[["s1_id", "cand_id"]].itertuples(index=False, name=None)))
    selected = _selected_pairs(preds, params, truth)
    pred_set = set(map(tuple, preds[["s1_id", "cand_id"]].itertuples(index=False, name=None)))
    true_set = {(str(s1), str(c)) for s1, ids in truth.items() for c in ids}

    # False positives: predicted but not true. Highest raw model probability first.
    fp_mask = preds.apply(lambda r: (str(r.s1_id), str(r.cand_id)) in selected and
                          (str(r.s1_id), str(r.cand_id)) not in true_set, axis=1)
    fp = preds.loc[fp_mask].sort_values("prob", ascending=False).head(100).copy()
    fp = _enrich(fp, records, truth)
    fp.to_csv(out_dir / f"{run_id}_fp.csv", index=False)

    # False negatives: every true pair not selected. Distinguish blocking miss from selection miss.
    missed = []
    for s1, cand in sorted(true_set):
        if (s1, cand) in selected:
            continue
        in_candidates = (s1, cand) in candidate_set
        row = preds[(preds["s1_id"] == s1) & (preds["cand_id"] == cand)]
        if not row.empty:
            rec = row.iloc[0].copy()
            rec["miss_type"] = "in candidates but not selected" if in_candidates else "not in candidates"
            missed.append(rec)
        else:
            missed.append(pd.Series({
                "s1_id": s1,
                "cand_id": cand,
                "prob": np.nan,
                "rank_within_s1": np.nan,
                "miss_type": "not in candidates" if not in_candidates else "in candidates but not selected",
            }))

    fn = pd.DataFrame(missed)
    if not fn.empty:
        fn["miss_type_order"] = fn["miss_type"].map({
            "not in candidates": 0,
            "in candidates but not selected": 1,
        }).fillna(2)
        fn = fn.sort_values(
            ["miss_type_order", "prob"],
            ascending=[True, True],
            na_position="first",
        ).head(100).drop(columns="miss_type_order")
        fn = _enrich(fn, records, truth)
        # _enrich drops miss_type, so restore it by key.
        miss_map = {(str(r.s1_id), str(r.cand_id)): r.miss_type for r in missed}
        fn["miss_type"] = [miss_map.get((s, c), "") for s, c in zip(fn.s1_id, fn.cand_id)]
    else:
        fn = pd.DataFrame(columns=FN_COLUMNS)

    fn = fn[FN_COLUMNS]
    fn.to_csv(out_dir / f"{run_id}_fn.csv", index=False)

    _print_loss_breakdown(preds, selected, truth)
    print(f"Wrote: {out_dir / f'{run_id}_fp.csv'}")
    print(f"Wrote: {out_dir / f'{run_id}_fn.csv'}")


def _f05(pred: set[str], true: set[str]) -> float:
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    fp = len(pred - true)
    fn = len(true - pred)
    if tp == 0:
        return 0.0
    return float(1.25 * tp / (1.25 * tp + 0.25 * fn + fp))


def _print_loss_breakdown(preds: pd.DataFrame, selected: set[tuple[str, str]], truth: dict[str, list[str]]) -> None:
    s1_ids = set(preds["s1_id"].astype(str)) | set(truth)
    singleton_loss = 0.0
    non_singleton_empty_loss = 0.0
    partial_loss = 0.0

    singleton_count = 0
    non_singleton_empty_count = 0
    partial_count = 0

    for s1 in s1_ids:
        true = set(truth.get(s1, []))
        pred = {c for e, c in selected if e == s1}
        loss = 1.0 - _f05(pred, true)

        if not true and pred:
            singleton_loss += loss
            singleton_count += 1
        elif true and not pred:
            non_singleton_empty_loss += loss
            non_singleton_empty_count += 1
        elif true and pred and pred != true:
            partial_loss += loss
            partial_count += 1

    print("\nEntity-level loss breakdown")
    print(f"Singletons we matched:       {singleton_count:6d} entities, {singleton_loss:.6f} points lost")
    print(f"Non-singletons left empty:   {non_singleton_empty_count:6d} entities, {non_singleton_empty_loss:.6f} points lost")
    print(f"Partial sets:                 {partial_count:6d} entities, {partial_loss:.6f} points lost")
    print(f"Total reported loss: {(singleton_loss + non_singleton_empty_loss + partial_loss):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write AR-6 FP/FN error sheets")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    params = _read_params(root, args.run_id)
    preds = _read_predictions(root, args.run_id)
    candidates = _read_candidates(root)
    truth = _truth(root)
    records = _read_records(root)

    print(f"Run: {args.run_id}")
    print(f"Postprocess method: {params.get('method', 'threshold')}")
    _write_error_sheets(root, args.run_id, preds, candidates, truth, params, records)


if __name__ == "__main__":
    main()

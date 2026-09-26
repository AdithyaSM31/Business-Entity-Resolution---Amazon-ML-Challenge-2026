"""Decision layer: turns pair probabilities into one match list per S1. Owner: Arushi (AR-4, AR-5).

Steps, each kept only if it improves out-of-fold AND leave-one-country-out macro F0.5:
  1. isotonic calibration of the probabilities
  2. exclusivity: each S2/S3 record goes to at most one S1 (only if BH-1 confirms the ground truth obeys this)
  3. per-S1 set selection maximizing expected F0.5 (Poisson-binomial DP; choosing nothing scores prod(1 - p_i)),
     compared against a tuned global threshold (optionally one per source)
  4. optional "has any match" gate
Output: dict s1_id -> list of matched IDs, handed to io_utils.write_id_lists.
"""

"""Turn pair probabilities into entity matches and tune AR-4 selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics


_REQUIRED = {"s1_id", "cand_id", "prob"}
_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def _validate_preds(preds: pd.DataFrame) -> pd.DataFrame:
    missing = _REQUIRED - set(preds.columns)
    if missing:
        raise ValueError(f"preds missing columns: {sorted(missing)}")
    out = preds[["s1_id", "cand_id", "prob"]].copy()
    out["s1_id"] = out["s1_id"].astype(str)
    out["cand_id"] = out["cand_id"].astype(str)
    out["prob"] = pd.to_numeric(out["prob"], errors="coerce")
    out = out.dropna(subset=["prob"])
    if ((out["prob"] < 0) | (out["prob"] > 1)).any():
        raise ValueError("prob must be in [0, 1]")
    return out


def _source(cand_id: str) -> str:
    if cand_id.startswith("S2-"):
        return "s2"
    if cand_id.startswith("S3-"):
        return "s3"
    return "other"


def _apply_exclusive(preds: pd.DataFrame) -> pd.DataFrame:
    """Keep the unique highest-probability S1 for each candidate ID.

    If two or more rows tie at the maximum probability, all tied rows are
    removed as required by AR-4.
    """
    if preds.empty:
        return preds.copy()
    p = preds.copy()
    max_p = p.groupby("cand_id")["prob"].transform("max")
    at_max = p["prob"].eq(max_p)
    n_max = at_max.groupby(p["cand_id"]).transform("sum")
    return p.loc[at_max & n_max.eq(1)].copy()


def _thresholds(params: dict) -> tuple[float, bool, float, float, bool]:
    threshold = float(params.get("threshold", 0.5))
    per_source = bool(params.get("per_source", False))
    threshold_s2 = float(params.get("threshold_s2", threshold))
    threshold_s3 = float(params.get("threshold_s3", threshold))
    exclusive = bool(params.get("exclusive", False))
    for name, value in (("threshold", threshold), ("threshold_s2", threshold_s2), ("threshold_s3", threshold_s3)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")
    return threshold, per_source, threshold_s2, threshold_s3, exclusive


def select_matches(preds: pd.DataFrame, params: dict) -> dict[str, list[str]]:
    """Select every candidate above the configured threshold.

    With ``exclusive=True``, each candidate ID can belong to at most one S1:
    only a unique highest-probability row survives; ties for the maximum are
    dropped. Thresholding happens after this exclusivity step.
    """
    p = _validate_preds(preds)
    _, per_source, threshold_s2, threshold_s3, exclusive = _thresholds(params)
    if p.empty:
        return {}
    p = p.drop_duplicates(["s1_id", "cand_id"], keep="last")
    if exclusive:
        p = _apply_exclusive(p)
    if per_source:
        source = p["cand_id"].map(_source)
        threshold = np.where(source.eq("s2"), threshold_s2, np.where(source.eq("s3"), threshold_s3, float(params.get("threshold", 0.5))))
    else:
        threshold = float(params.get("threshold", 0.5))
    p = p.loc[p["prob"].to_numpy() >= threshold]
    if p.empty:
        return {}
    return p.groupby("s1_id", sort=False)["cand_id"].agg(list).to_dict()


def _score_selected(selected: dict[str, list[str]], truth: dict[str, list[str]]) -> float:
    return float(metrics.macro_f05(selected, truth))


def _score_threshold(p: pd.DataFrame, truth: dict[str, list[str]], threshold: float) -> float:
    selected = select_matches(p, {"threshold": threshold, "per_source": False, "exclusive": False})
    return _score_selected(selected, truth)


def _curve_for_thresholds(p: pd.DataFrame, truth: dict[str, list[str]], thresholds: np.ndarray, exclusive: bool, per_source: bool):
    """Return (best_params, curve) while keeping the common path simple.

    The 91-point curve is evaluated for the shared-threshold case. For
    per-source mode we evaluate the full S2 x S3 grid and print the best
    F0.5 at each S2 threshold; the returned parameters use the best pair.
    """
    base = _apply_exclusive(p) if exclusive else p
    if base.empty:
        default = {"threshold": 0.5, "per_source": per_source, "exclusive": exclusive}
        if per_source:
            default.update(threshold_s2=0.5, threshold_s3=0.5)
        return default, [(float(t), 1.0) for t in thresholds]

    truth_all = {str(k): list(v) for k, v in truth.items()}
    if not per_source:
        rows = []
        best = (-1.0, None)
        for t in thresholds:
            score = _score_threshold(base, truth_all, float(t))
            rows.append((float(t), score))
            if score > best[0]:
                best = (score, float(t))
        return {"threshold": best[1], "per_source": False, "exclusive": exclusive}, rows

    # Full independent threshold search for S2/S3. Each combination is scored
    # through the official metric, matching the required objective exactly.
    best_score = -1.0
    best_pair = (0.5, 0.5)
    curve = []
    for t2 in thresholds:
        row_best = -1.0
        for t3 in thresholds:
            params = {"threshold": 0.5, "per_source": True, "threshold_s2": float(t2), "threshold_s3": float(t3), "exclusive": exclusive}
            score = _score_selected(select_matches(base, params), truth_all)
            if score > row_best:
                row_best = score
            if score > best_score:
                best_score = score
                best_pair = (float(t2), float(t3))
        curve.append((float(t2), row_best))
    return {"threshold": 0.5, "per_source": True, "threshold_s2": best_pair[0], "threshold_s3": best_pair[1], "exclusive": exclusive}, curve


def tune(oof_preds: pd.DataFrame, truth: dict[str, list[str]]) -> dict:
    """Grid-search AR-4 selection parameters on out-of-fold predictions.

    Searches thresholds 0.05..0.95 for non-per-source selection and the full
    S2 x S3 threshold grid for per-source selection, with exclusivity on/off.
    The objective is exactly ``metrics.macro_f05``. The best F0.5 curve point
    for each mode is printed so the flatness of the optimum is visible.
    """
    p = _validate_preds(oof_preds)
    if p.empty:
        raise ValueError("Cannot tune postprocess parameters on empty predictions")

    results = []
    best_score = -1.0
    best_params = None
    for exclusive in (False, True):
        for per_source in (False, True):
            params, curve = _curve_for_thresholds(p, truth, _GRID, exclusive, per_source)
            score = _score_selected(select_matches(p, params), truth)
            results.append((exclusive, per_source, params, score, curve))
            print(f"\nF0.5 curve: exclusive={exclusive}, per_source={per_source}")
            print("threshold\tF0.5")
            for threshold, f05 in curve:
                print(f"{threshold:.2f}\t{f05:.6f}")
            if score > best_score:
                best_score = score
                best_params = params.copy()

    best_params = dict(best_params)
    best_params["cv_f05"] = float(best_score)
    print("\nBest AR-4 parameters:")
    print(json.dumps(best_params, indent=2))
    default_path = Path("models") / "postprocess.json"
    default_path.parent.mkdir(parents=True, exist_ok=True)
    with default_path.open("w", encoding="utf-8") as f:
        json.dump({k: v for k, v in best_params.items() if k != "cv_f05"}, f, indent=2)
    print(f"Saved params: {default_path}")
    return best_params


def save_params(params: dict, run_id: str, project_root: str | Path = ".") -> Path:
    path = Path(project_root) / "models" / str(run_id) / "postprocess.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in params.items() if k != "cv_f05"}
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path

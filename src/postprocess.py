"""Decision layer: turns pair probabilities into one match list per S1.

Owner: Arushi (AR-4, AR-5).

AR-4:
  1. Threshold selection from 0.05 to 0.95.
  2. Optional per-source thresholds for S2/S3.
  3. Optional exclusivity: each candidate goes to at most one S1.
  4. Tied highest probabilities are dropped.
  5. Select parameters using Macro F0.5 on OOF predictions.

AR-5:
  1. Isotonic calibration using cross-fitted OOF predictions.
  2. Expected-F0.5 per-S1 selection using Poisson-binomial DP.
  3. Optional exclusivity.
  4. Optional has-match LogisticRegression gate.
  5. Compare threshold, expected-F0.5 and expected-F0.5+gate.
  6. Save the selected parameters.

Output:
  dict s1_id -> list of matched IDs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import metrics


_REQUIRED = {"s1_id", "cand_id", "prob"}
_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


# ============================================================
# AR-4: EXISTING VALIDATION / SELECTION CODE
# ============================================================

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

    for name, value in (
        ("threshold", threshold),
        ("threshold_s2", threshold_s2),
        ("threshold_s3", threshold_s3),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    return threshold, per_source, threshold_s2, threshold_s3, exclusive


def select_matches(preds: pd.DataFrame, params: dict) -> dict[str, list[str]]:
    """Select matches using AR-4 thresholding or AR-5 expected F0.5."""
    method = str(params.get("method", "threshold")).lower()

    if method == "expected_f":
        return _expected_f_select(
            preds,
            exclusive=bool(params.get("exclusive", False)),
            max_candidates=int(params.get("max_candidates", 50)),
            miss_rate=float(params.get("miss_rate", 0.0)),
        )

    if method == "expected_f_gate":
        selected = _expected_f_select(
            preds,
            exclusive=bool(params.get("exclusive", False)),
            max_candidates=int(params.get("max_candidates", 50)),
            miss_rate=float(params.get("miss_rate", 0.0)),
        )

        return selected

    # ---------------- AR-4 original behavior ----------------
    p = _validate_preds(preds)

    _, per_source, threshold_s2, threshold_s3, exclusive = _thresholds(params)

    if p.empty:
        return {}

    p = p.drop_duplicates(["s1_id", "cand_id"], keep="last")

    if exclusive:
        p = _apply_exclusive(p)

    if per_source:
        source = p["cand_id"].map(_source)
        threshold = np.where(
            source.eq("s2"),
            threshold_s2,
            np.where(
                source.eq("s3"),
                threshold_s3,
                float(params.get("threshold", 0.5)),
            ),
        )
    else:
        threshold = float(params.get("threshold", 0.5))

    p = p.loc[p["prob"].to_numpy() >= threshold]

    if p.empty:
        return {}

    return p.groupby("s1_id", sort=False)["cand_id"].agg(list).to_dict()


def _score_selected(
    selected: dict[str, list[str]],
    truth: dict[str, list[str]],
) -> float:
    return float(metrics.macro_f05(selected, truth))


def _score_threshold(
    p: pd.DataFrame,
    truth: dict[str, list[str]],
    threshold: float,
) -> float:
    selected = select_matches(
        p,
        {
            "threshold": threshold,
            "per_source": False,
            "exclusive": False,
        },
    )
    return _score_selected(selected, truth)


def _curve_for_thresholds(
    p: pd.DataFrame,
    truth: dict[str, list[str]],
    thresholds: np.ndarray,
    exclusive: bool,
    per_source: bool,
):
    """AR-4 threshold search."""

    base = _apply_exclusive(p) if exclusive else p

    if base.empty:
        default = {
            "threshold": 0.5,
            "per_source": per_source,
            "exclusive": exclusive,
        }

        if per_source:
            default.update(
                threshold_s2=0.5,
                threshold_s3=0.5,
            )

        return default, [(float(t), 1.0) for t in thresholds]

    truth_all = {str(k): list(v) for k, v in truth.items()}

    if not per_source:
        rows = []
        best = (-1.0, None)

        for t in thresholds:
            score = _score_threshold(
                base,
                truth_all,
                float(t),
            )

            rows.append((float(t), score))

            if score > best[0]:
                best = (score, float(t))

        return {
            "threshold": best[1],
            "per_source": False,
            "exclusive": exclusive,
        }, rows

    best_score = -1.0
    best_pair = (0.5, 0.5)
    curve = []

    for t2 in thresholds:
        row_best = -1.0

        for t3 in thresholds:
            params = {
                "threshold": 0.5,
                "per_source": True,
                "threshold_s2": float(t2),
                "threshold_s3": float(t3),
                "exclusive": exclusive,
            }

            score = _score_selected(
                select_matches(base, params),
                truth_all,
            )

            if score > row_best:
                row_best = score

            if score > best_score:
                best_score = score
                best_pair = (float(t2), float(t3))

        curve.append((float(t2), row_best))

    return {
        "threshold": 0.5,
        "per_source": True,
        "threshold_s2": best_pair[0],
        "threshold_s3": best_pair[1],
        "exclusive": exclusive,
    }, curve


# ============================================================
# AR-5: CALIBRATION
# ============================================================

def _truth_labels(
    preds: pd.DataFrame,
    truth: dict[str, list[str]],
) -> np.ndarray:
    """Create pair-level binary labels from the S1 -> true candidates map."""
    true_sets = {
        str(s1): {str(c) for c in candidates}
        for s1, candidates in truth.items()
    }

    return np.array(
        [
            int(cand in true_sets.get(s1, set()))
            for s1, cand in zip(preds["s1_id"], preds["cand_id"])
        ],
        dtype=int,
    )


def fit_isotonic_oof(
    oof_preds: pd.DataFrame,
    truth: dict[str, list[str]],
) -> pd.DataFrame:
    """Cross-fitted isotonic calibration on OOF predictions.

    Each fold's calibrator is fitted only on the other folds.
    """
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import brier_score_loss

    p = _validate_preds(oof_preds).copy()

    if "fold" not in oof_preds.columns:
        raise ValueError(
            "AR-5 OOF calibration requires a 'fold' column."
        )

    p["fold"] = oof_preds["fold"].to_numpy()

    y = _truth_labels(p, truth)
    raw = p["prob"].to_numpy(dtype=float)
    calibrated = np.empty(len(p), dtype=float)

    folds = list(pd.unique(p["fold"]))

    for fold in folds:
        train_mask = p["fold"].to_numpy() != fold
        test_mask = p["fold"].to_numpy() == fold

        x_train = raw[train_mask]
        y_train = y[train_mask]

        if len(np.unique(y_train)) < 2:
            calibrated[test_mask] = raw[test_mask]
            continue

        calibrator = IsotonicRegression(
            out_of_bounds="clip"
        )

        calibrator.fit(x_train, y_train)
        calibrated[test_mask] = calibrator.predict(raw[test_mask])

    before = brier_score_loss(y, raw)
    after = brier_score_loss(y, calibrated)

    print("\nAR-5 Isotonic calibration")
    print(f"Brier before: {before:.6f}")
    print(f"Brier after:  {after:.6f}")

    p["prob_calibrated"] = np.clip(calibrated, 0.0, 1.0)

    return p


def fit_isotonic_all(
    train_preds: pd.DataFrame,
    truth: dict[str, list[str]],
):
    """Fit the final isotonic calibrator on all available training data."""
    from sklearn.isotonic import IsotonicRegression

    p = _validate_preds(train_preds)
    y = _truth_labels(p, truth)

    if len(np.unique(y)) < 2:
        return None

    calibrator = IsotonicRegression(
        out_of_bounds="clip"
    )

    calibrator.fit(
        p["prob"].to_numpy(dtype=float),
        y,
    )

    return calibrator


def apply_calibrator(
    preds: pd.DataFrame,
    calibrator,
) -> pd.DataFrame:
    """Apply a fitted isotonic calibrator."""
    p = _validate_preds(preds).copy()

    if calibrator is None:
        p["prob_calibrated"] = p["prob"]
        return p

    p["prob_calibrated"] = np.clip(
        calibrator.predict(
            p["prob"].to_numpy(dtype=float)
        ),
        0.0,
        1.0,
    )

    return p


# ============================================================
# AR-5: POISSON-BINOMIAL EXPECTED F0.5
# ============================================================

def _poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    """Return P(A=a) for independent Bernoulli variables."""
    probabilities = np.asarray(probabilities, dtype=float)

    dp = np.array([1.0], dtype=float)

    for p in probabilities:
        next_dp = np.zeros(len(dp) + 1, dtype=float)
        next_dp[:-1] += dp * (1.0 - p)
        next_dp[1:] += dp * p
        dp = next_dp

    return dp

def _expected_f(
    selected_p: np.ndarray,
    remaining_p: np.ndarray,
    miss_rate: float = 0.0,
) -> float:
    """Expected F0.5 for one selected set.

    A = number of true matches among selected candidates.
    B = number of true matches among remaining candidates.

    For k > 0:
        F0.5 = 1.25 A / (0.25(A+B) + k)

    k = 0:
        expected score = probability that there are no true matches.

    miss_rate optionally adds an independent miss probability to B.
    """
    selected_p = np.asarray(selected_p, dtype=float)
    remaining_p = np.asarray(remaining_p, dtype=float)

    k = len(selected_p)

    if k == 0:
        if len(remaining_p) == 0:
            return 1.0

        return float(
            np.prod(1.0 - remaining_p)
        )

    a_dist = _poisson_binomial(selected_p)
    b_dist = _poisson_binomial(remaining_p)

    if miss_rate > 0 and len(b_dist) > 0:
        miss_dist = _poisson_binomial(
            np.full(
                len(remaining_p),
                float(miss_rate),
            )
        )

        # Add misses as additional Bernoulli contributions.
        combined = np.zeros(
            len(b_dist) + len(miss_dist) - 1,
            dtype=float,
        )

        for i, pa in enumerate(b_dist):
            combined[i:i + len(miss_dist)] += pa * miss_dist

        b_dist = combined

    expected = 0.0

    for a, pa in enumerate(a_dist):
        if pa == 0:
            continue

        for b, pb in enumerate(b_dist):
            if pb == 0:
                continue

            denominator = 0.25 * (a + b) + k

            if denominator <= 0:
                continue

            f = 1.25 * a / denominator
            expected += pa * pb * f

    return float(expected)


def _expected_f_select_one(
    group: pd.DataFrame,
    max_candidates: int = 50,
    miss_rate: float = 0.0,
) -> list[str]:
    """Select the best prefix of candidates sorted by probability."""
    if group.empty:
        return []

    ordered = group.sort_values(
        "prob_calibrated",
        ascending=False,
    ).reset_index(drop=True)

    ordered = ordered.head(max_candidates)

    probs = ordered["prob_calibrated"].to_numpy(
        dtype=float
    )

    candidates = ordered["cand_id"].astype(str).tolist()

    best_score = -1.0
    best_k = 0

    for k in range(len(probs) + 1):
        selected_p = probs[:k]
        remaining_p = probs[k:]

        score = _expected_f(
            selected_p,
            remaining_p,
            miss_rate=miss_rate,
        )

        if score > best_score:
            best_score = score
            best_k = k

    return candidates[:best_k]


def _expected_f_select(
    preds: pd.DataFrame,
    exclusive: bool = False,
    max_candidates: int = 50,
    miss_rate: float = 0.0,
) -> dict[str, list[str]]:
    """Expected-F0.5 selection independently for each S1."""

    p = _validate_preds(preds).copy()

    if "prob_calibrated" not in preds.columns:
        p["prob_calibrated"] = p["prob"]

    p = p.drop_duplicates(
        ["s1_id", "cand_id"],
        keep="last",
    )

    if exclusive:
        max_p = p.groupby("cand_id")["prob_calibrated"].transform("max")
        at_max = p["prob_calibrated"].eq(max_p)
        n_max = at_max.groupby(p["cand_id"]).transform("sum")
        p = p.loc[at_max & n_max.eq(1)].copy()

    if p.empty:
        return {}

    result = {}

    for s1_id, group in p.groupby("s1_id", sort=False):
        selected = _expected_f_select_one(
            group,
            max_candidates=max_candidates,
            miss_rate=miss_rate,
        )

        # IMPORTANT:
        # Keep S1 even when the optimal decision is to select nothing.
        result[str(s1_id)] = selected

    return result


# ============================================================
# AR-5: OPTIONAL HAS-MATCH GATE
# ============================================================

def _build_gate_features(
    preds: pd.DataFrame,
) -> pd.DataFrame:
    """Build one feature row per S1."""
    p = _validate_preds(preds).copy()

    if "prob_calibrated" not in p.columns:
        p["prob_calibrated"] = p["prob"]

    rows = []

    for s1_id, group in p.groupby(
        "s1_id",
        sort=False,
    ):
        probs = np.sort(
            group["prob_calibrated"].to_numpy(
                dtype=float
            )
        )[::-1]

        max_p = float(probs[0]) if len(probs) else 0.0
        second_p = float(probs[1]) if len(probs) > 1 else 0.0

        row = {
            "s1_id": str(s1_id),
            "max_p": max_p,
            "second_p": second_p,
            "count_p_gt_05": float(
                np.sum(probs > 0.5)
            ),
        }

        if "fc_n_close" in group.columns:
            row["fc_n_close"] = float(
                pd.to_numeric(
                    group["fc_n_close"],
                    errors="coerce",
                )
                .fillna(0)
                .max()
            )

        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "s1_id",
                "max_p",
                "second_p",
                "count_p_gt_05",
            ]
        )

    return pd.DataFrame(rows)


def _gate_predictions(
    preds: pd.DataFrame,
    truth: dict[str, list[str]],
    gate_threshold: float = 0.5,
) -> dict[str, list[str]]:
    """Cross-fitted LogisticRegression has-any-match gate.

    The gate only decides whether an S1 is allowed to have matches.
    The actual candidate selection remains AR-4 threshold selection.
    """
    from sklearn.linear_model import LogisticRegression

    if "fold" not in preds.columns:
        raise ValueError(
            "AR-5 gate requires a 'fold' column."
        )

    features = _build_gate_features(preds)

    true_has_match = {
        str(s1): int(len(candidates) > 0)
        for s1, candidates in truth.items()
    }

    features["label"] = features["s1_id"].map(
        true_has_match
    ).fillna(0).astype(int)

    feature_cols = [
        "max_p",
        "second_p",
        "count_p_gt_05",
    ]

    if "fc_n_close" in features.columns:
        feature_cols.append("fc_n_close")

    fold_map = (
        preds[["s1_id", "fold"]]
        .drop_duplicates("s1_id")
    )

    features = features.merge(
        fold_map,
        on="s1_id",
        how="left",
    )

    gate_probability = pd.Series(
        0.0,
        index=features.index,
        dtype=float,
    )

    for fold in pd.unique(features["fold"]):
        train_mask = features["fold"] != fold
        test_mask = features["fold"] == fold

        train = features.loc[train_mask]
        test = features.loc[test_mask]

        if train.empty or test.empty:
            continue

        if train["label"].nunique() < 2:
            gate_probability.loc[test.index] = float(
                train["label"].mean()
            )
            continue

        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        model.fit(
            train[feature_cols],
            train["label"],
        )

        gate_probability.loc[test.index] = (
            model.predict_proba(
                test[feature_cols]
            )[:, 1]
        )

    allowed = set(
        features.loc[
            gate_probability >= gate_threshold,
            "s1_id",
        ].astype(str)
    )

    # AR-4 threshold selection remains the candidate selector.
    base_params = {
        "threshold": 0.5,
        "per_source": False,
        "exclusive": False,
    }

    selected = select_matches(
        preds,
        base_params,
    )

    return {
        s1: candidates
        for s1, candidates in selected.items()
        if s1 in allowed
    }


# ============================================================
# AR-5: METHOD SELECTION
# ============================================================

def _select_by_method(
    preds: pd.DataFrame,
    method: str,
    params: dict,
) -> dict[str, list[str]]:
    """Run one AR-5 selection method."""
    method = str(method).lower()

    if method == "threshold":
        return select_matches(
            preds,
            params,
        )

    if method == "expected_f":
        return _expected_f_select(
            preds,
            exclusive=bool(
                params.get("exclusive", False)
            ),
            max_candidates=int(
                params.get("max_candidates", 50)
            ),
            miss_rate=float(
                params.get("miss_rate", 0.0)
            ),
        )

    if method == "expected_f_gate":
        selected = _expected_f_select(
            preds,
            exclusive=bool(
                params.get("exclusive", False)
            ),
            max_candidates=int(
                params.get("max_candidates", 50)
            ),
            miss_rate=float(
                params.get("miss_rate", 0.0)
            ),
        )

        # Gate is applied separately in tune_ar5 because
        # it needs cross-fitted gate predictions.
        return selected

    raise ValueError(
        f"Unknown AR-5 selection method: {method}"
    )


# ============================================================
# AR-5: TUNING
# ============================================================

def _tune_ar5_threshold(
    preds: pd.DataFrame,
    truth: dict[str, list[str]],
) -> tuple[dict, float]:
    """Tune the original AR-4 threshold selector for AR-5 comparison."""
    best_params = None
    best_score = -1.0

    for exclusive in (False, True):
        for per_source in (False, True):
            params, _ = _curve_for_thresholds(
                preds,
                truth,
                _GRID,
                exclusive,
                per_source,
            )

            selected = select_matches(
                preds,
                params,
            )

            score = _score_selected(
                selected,
                truth,
            )

            if score > best_score:
                best_score = score
                best_params = dict(params)

    return best_params, float(best_score)


def tune_ar5(
    oof_preds: pd.DataFrame,
    truth: dict[str, list[str]],
    loco_preds: pd.DataFrame | None = None,
    loco_truth: dict[str, list[str]] | None = None,
    run_id: str | None = None,
    use_exclusive: bool = False,
    gate_threshold: float = 0.5,
    max_candidates: int = 50,
    miss_rate: float = 0.0,
) -> dict:
    """Run the complete AR-5 comparison.

    Methods compared:
      1. Tuned threshold baseline.
      2. Expected F0.5.
      3. Expected F0.5 + has-match gate.

    OOF is the primary tuning/evaluation set.
    LOCO is optional and is only used for comparison/reporting.

    ``use_exclusive`` should only be enabled when BH-1 confirms that
    the ground truth obeys the one-candidate-to-one-S1 rule.
    """
    p = _validate_preds(oof_preds).copy()

    if "fold" not in oof_preds.columns:
        raise ValueError(
            "AR-5 requires OOF predictions with a 'fold' column."
        )

    p["fold"] = oof_preds["fold"].to_numpy()

    # --------------------------------------------------------
    # 1. Calibrate OOF probabilities
    # --------------------------------------------------------
    calibrated = fit_isotonic_oof(
        p,
        truth,
    )

    # Preserve fold after calibration.
    calibrated["fold"] = p["fold"].to_numpy()

    # --------------------------------------------------------
    # 2. Threshold baseline
    # --------------------------------------------------------
    threshold_params, threshold_score = _tune_ar5_threshold(
        calibrated.rename(
            columns={"prob_calibrated": "prob"}
        ),
        truth,
    )

    threshold_params["method"] = "threshold"

    threshold_selected = select_matches(
        calibrated.rename(
            columns={"prob_calibrated": "prob"}
        ),
        threshold_params,
    )

    threshold_score = _score_selected(
        threshold_selected,
        truth,
    )

    # --------------------------------------------------------
    # 3. Expected F0.5
    # --------------------------------------------------------
    expected_params = {
        "method": "expected_f",
        "exclusive": bool(use_exclusive),
        "max_candidates": int(max_candidates),
        "miss_rate": float(miss_rate),
    }

    expected_selected = _expected_f_select(
        calibrated,
        exclusive=use_exclusive,
        max_candidates=max_candidates,
        miss_rate=miss_rate,
    )

    expected_score = _score_selected(
        expected_selected,
        truth,
    )

    # --------------------------------------------------------
    # 4. Expected F0.5 + gate
    # --------------------------------------------------------
    gate_params = dict(expected_params)
    gate_params["method"] = "expected_f_gate"
    gate_params["gate_threshold"] = float(
        gate_threshold
    )

    expected_gate_selected = _expected_f_select(
        calibrated,
        exclusive=use_exclusive,
        max_candidates=max_candidates,
        miss_rate=miss_rate,
    )

    # Build gate features and cross-fit gate.
    gate_features = _build_gate_features(
        calibrated
    )

    true_has_match = {
        str(s1): int(len(candidates) > 0)
        for s1, candidates in truth.items()
    }

    gate_features["label"] = gate_features[
        "s1_id"
    ].map(true_has_match).fillna(0).astype(int)

    gate_features = gate_features.merge(
        calibrated[["s1_id", "fold"]].drop_duplicates(
            "s1_id"
        ),
        on="s1_id",
        how="left",
    )

    from sklearn.linear_model import LogisticRegression

    feature_cols = [
        "max_p",
        "second_p",
        "count_p_gt_05",
    ]

    if "fc_n_close" in gate_features.columns:
        feature_cols.append("fc_n_close")

    gate_probability = np.zeros(
        len(gate_features),
        dtype=float,
    )

    for fold in pd.unique(gate_features["fold"]):
        train_mask = gate_features["fold"] != fold
        test_mask = gate_features["fold"] == fold

        train = gate_features.loc[train_mask]
        test = gate_features.loc[test_mask]

        if train.empty or test.empty:
            continue

        if train["label"].nunique() < 2:
            gate_probability[
                gate_features.index.isin(test.index)
            ] = float(train["label"].mean())
            continue

        gate_model = LogisticRegression(
            max_iter=1000,
            random_state=42,
        )

        gate_model.fit(
            train[feature_cols],
            train["label"],
        )

        gate_probability[
            gate_features.index.isin(test.index)
        ] = gate_model.predict_proba(
            test[feature_cols]
        )[:, 1]

    allowed_s1 = set(
        gate_features.loc[
            gate_probability >= gate_threshold,
            "s1_id",
        ].astype(str)
    )

    expected_gate_selected = {
        s1: candidates
        for s1, candidates in expected_gate_selected.items()
        if s1 in allowed_s1
    }

    expected_gate_score = _score_selected(
        expected_gate_selected,
        truth,
    )

    # --------------------------------------------------------
    # 5. Print comparison
    # --------------------------------------------------------
    comparison = pd.DataFrame(
        [
            {
                "method": "threshold",
                "OOF_macro_F0.5": threshold_score,
            },
            {
                "method": "expected_f",
                "OOF_macro_F0.5": expected_score,
            },
            {
                "method": "expected_f_gate",
                "OOF_macro_F0.5": expected_gate_score,
            },
        ]
    )

    print("\nAR-5 OOF comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # --------------------------------------------------------
    # 6. Select best OOF method
    # --------------------------------------------------------
    scores = {
        "threshold": threshold_score,
        "expected_f": expected_score,
        "expected_f_gate": expected_gate_score,
    }

    best_method = max(
        scores,
        key=scores.get,
    )

    if best_method == "threshold":
        best_params = dict(threshold_params)

    elif best_method == "expected_f":
        best_params = dict(expected_params)

    else:
        best_params = dict(gate_params)

    best_params["cv_f05"] = float(
        scores[best_method]
    )

    # --------------------------------------------------------
    # 7. Optional LOCO comparison
    # --------------------------------------------------------
    if loco_preds is not None and loco_truth is not None:
        loco = _validate_preds(
            loco_preds
        ).copy()

        if "prob_calibrated" not in loco.columns:
            loco["prob_calibrated"] = loco["prob"]

        loco_threshold_selected = select_matches(
            loco,
            threshold_params,
        )

        loco_threshold_score = _score_selected(
            loco_threshold_selected,
            loco_truth,
        )

        loco_expected_selected = _expected_f_select(
            loco,
            exclusive=use_exclusive,
            max_candidates=max_candidates,
            miss_rate=miss_rate,
        )

        loco_expected_score = _score_selected(
            loco_expected_selected,
            loco_truth,
        )

        print("\nAR-5 LOCO comparison:")
        print(
            f"threshold:       {loco_threshold_score:.6f}"
        )
        print(
            f"expected_f:      {loco_expected_score:.6f}"
        )

    # --------------------------------------------------------
    # 8. Save parameters
    # --------------------------------------------------------
    save_payload = {
        k: v
        for k, v in best_params.items()
        if k != "cv_f05"
    }

    if run_id is not None:
        path = save_params(
            save_payload,
            run_id,
        )
    else:
        path = (
            Path("models")
            / "postprocess.json"
        )
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                save_payload,
                f,
                indent=2,
            )

    print("\nBest AR-5 parameters:")
    print(
        json.dumps(
            best_params,
            indent=2,
        )
    )

    print(f"Saved params: {path}")

    return best_params


# ============================================================
# AR-4 TUNE — PRESERVED
# ============================================================

def tune(
    oof_preds: pd.DataFrame,
    truth: dict[str, list[str]],
    ar5: bool = False,
    **ar5_kwargs,
) -> dict:
    """Tune post-processing.

    Default behavior is the original AR-4 implementation.

    Set ``ar5=True`` to run the AR-5 calibration/expected-F0.5
    extension without changing the existing AR-4 behavior.
    """
    if ar5:
        return tune_ar5(
            oof_preds,
            truth,
            **ar5_kwargs,
        )

    p = _validate_preds(oof_preds)

    if p.empty:
        raise ValueError(
            "Cannot tune postprocess parameters on empty predictions"
        )

    results = []
    best_score = -1.0
    best_params = None

    for exclusive in (False, True):
        for per_source in (False, True):
            params, curve = _curve_for_thresholds(
                p,
                truth,
                _GRID,
                exclusive,
                per_source,
            )

            score = _score_selected(
                select_matches(p, params),
                truth,
            )

            results.append(
                (
                    exclusive,
                    per_source,
                    params,
                    score,
                    curve,
                )
            )

            print(
                f"\nF0.5 curve: "
                f"exclusive={exclusive}, "
                f"per_source={per_source}"
            )

            print("threshold\tF0.5")

            for threshold, f05 in curve:
                print(
                    f"{threshold:.2f}\t{f05:.6f}"
                )

            if score > best_score:
                best_score = score
                best_params = params.copy()

    best_params = dict(best_params)
    best_params["cv_f05"] = float(best_score)

    print("\nBest AR-4 parameters:")
    print(
        json.dumps(
            best_params,
            indent=2,
        )
    )

    default_path = (
        Path("models")
        / "postprocess.json"
    )

    default_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with default_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                k: v
                for k, v in best_params.items()
                if k != "cv_f05"
            },
            f,
            indent=2,
        )

    print(
        f"Saved params: {default_path}"
    )

    return best_params


def save_params(
    params: dict,
    run_id: str,
    project_root: str | Path = ".",
) -> Path:
    path = (
        Path(project_root)
        / "models"
        / str(run_id)
        / "postprocess.json"
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        k: v
        for k, v in params.items()
        if k != "cv_f05"
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )

    return path
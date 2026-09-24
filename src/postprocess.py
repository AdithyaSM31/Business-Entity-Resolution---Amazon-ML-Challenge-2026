"""Decision layer: turns pair probabilities into one match list per S1. Owner: Arushi (AR-4, AR-5).

Steps, each kept only if it improves out-of-fold AND leave-one-country-out macro F0.5:
  1. isotonic calibration of the probabilities
  2. exclusivity: each S2/S3 record goes to at most one S1 (only if BH-1 confirms the ground truth obeys this)
  3. per-S1 set selection maximizing expected F0.5 (Poisson-binomial DP; choosing nothing scores prod(1 - p_i)),
     compared against a tuned global threshold (optionally one per source)
  4. optional "has any match" gate
Output: dict s1_id -> list of matched IDs, handed to io_utils.write_id_lists.
"""
import pandas as pd


def select_matches(preds: pd.DataFrame, params: dict) -> dict[str, list[str]]:
    raise NotImplementedError("AR-4")


def tune(oof_preds: pd.DataFrame, truth: dict[str, list[str]]) -> dict:
    raise NotImplementedError("AR-4")

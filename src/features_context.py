"""Context and competition features. Owner: Siva (SI-4, SI-5).

fc_*  (SI-4, from blocking only): name/address frequency (chains, malls), near-duplicate count per S1,
      rank within the S1's candidates, reverse rank (rank of this S1 among the S1s that list this record),
      gap to the best candidate, pass flags and pass scores.
      Output: cache/feat_ctx_{split}.parquet
fc2_* (SI-5, from stage-1 predictions; on train these MUST be the out-of-fold predictions):
      rank and margin by probability, mutual best, best competing probability, cross-source support
      (max over the S1's other candidates a of p(s1, a) * sim(a, cand)).
      Output: cache/feat_ctx2_{split}_{run_id}.parquet
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from . import config
from . import io_utils
from .blocking import PASSES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_records(split: str) -> pd.DataFrame:
    """Load the columns we need from cache/records.parquet (fall back to raw TSVs)."""
    recs_path = config.cache_path("records.parquet")
    want = ["split", "source", "entity_id", "name_core", "addr_norm",
            "addr_numbers", "postcode"]
    if recs_path.exists():
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(recs_path).names)
        cols = [c for c in want if c in avail]
        df = pd.read_parquet(recs_path, columns=cols)
        df = df[df["split"] == split].copy()
    else:
        # Fallback: only name_core available (lowercased business_name)
        frames = []
        for src in config.SOURCES:
            raw = io_utils.read_source(split, src)
            raw["source"] = np.int8(src)
            raw["split"] = split
            raw["name_core"] = raw["business_name"].str.lower().fillna("")
            frames.append(raw[["split", "source", "entity_id", "name_core"]])
        df = pd.concat(frames, ignore_index=True)

    # Ensure safe defaults for optional columns
    for col in ["addr_norm", "postcode"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)
    if "name_core" not in df.columns:
        df["name_core"] = ""
    else:
        df["name_core"] = df["name_core"].fillna("").astype(str)
    if "addr_numbers" not in df.columns:
        df["addr_numbers"] = [[] for _ in range(len(df))]
    else:
        df["addr_numbers"] = df["addr_numbers"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    df["source"] = df["source"].astype(np.int8)
    return df


def _first(lst: list) -> str:
    """Return first element of a list, or '' if empty."""
    return lst[0] if lst else ""


# ---------------------------------------------------------------------------
# SI-4: main builder
# ---------------------------------------------------------------------------

def build_context_features(split: str) -> pd.DataFrame:
    """Compute context features for every candidate pair in candidates_{split}.parquet.

    Reads:
      cache/candidates_{split}.parquet   (the blocking output)
      cache/records.parquet              (or raw TSVs as fallback)

    Writes:
      cache/feat_ctx_{split}.parquet

    Feature catalogue
    -----------------
    fc_name_freq_s1   : records in this split sharing the exact name_core as S1
    fc_name_freq_cand : records in this split sharing the exact name_core as cand
    fc_addr_freq_s1   : distinct name_core values co-located at S1's address key
    fc_addr_freq_cand : distinct name_core values co-located at cand's address key
    fc_n_cands        : total candidates listed for this S1
    fc_n_close        : candidates for this S1 with bs_tfidf_name >= 0.8
    fc_rank           : prune_rank of this pair (0 = best)
    fc_gap_best       : best prune_score for this S1 minus this pair's prune_score
    fc_rev_rank       : rank of this S1 among all S1s that share this cand_id (by prune_score, 0=best)
    fc_rev_n          : number of S1s that list this cand_id as a candidate
    fc_n_passes       : number of blocking passes that retrieved this pair
    fc_cand_source    : 2 or 3
    fc_blk_{pass}     : 0/1 float – whether this pass found the pair
    fc_bs_{pass}      : raw pass score (NaN when pass did not find this pair)
    """
    t0 = time.perf_counter()
    print(f"\n=== build_context_features(split={split!r}) ===")

    # ------------------------------------------------------------------ #
    # 1.  Load candidates
    # ------------------------------------------------------------------ #
    cands_path = config.cache_path(f"candidates_{split}.parquet")
    if not cands_path.exists():
        raise FileNotFoundError(
            f"candidates_{split}.parquet not found. "
            f"Run blocking.generate_candidates('{split}') first."
        )
    cands = pd.read_parquet(cands_path)
    print(f"  candidates: {len(cands):,} pairs")

    # ------------------------------------------------------------------ #
    # 2.  Load records
    # ------------------------------------------------------------------ #
    recs = _load_records(split)
    print(f"  records:    {len(recs):,} rows")
    rec_idx = recs.set_index("entity_id")

    # ------------------------------------------------------------------ #
    # 3.  fc_name_freq_* : exact name_core frequency across the split
    # ------------------------------------------------------------------ #
    name_freq = recs.groupby("name_core")["entity_id"].count().rename("freq")

    s1_name_core   = rec_idx["name_core"].reindex(cands["s1_id"]).values
    cand_name_core = rec_idx["name_core"].reindex(cands["cand_id"]).values

    fc_name_freq_s1   = (
        pd.Series(s1_name_core).map(name_freq).fillna(1).astype(np.float32).values
    )
    fc_name_freq_cand = (
        pd.Series(cand_name_core).map(name_freq).fillna(1).astype(np.float32).values
    )

    # ------------------------------------------------------------------ #
    # 4.  fc_addr_freq_* : distinct name_core values at the same address key
    #     Primary key: (postcode, first addr_numbers entry)  when both non-empty
    #     Fallback key: addr_norm  (coarser; skipped if also empty)
    # ------------------------------------------------------------------ #
    postcode_s  = recs["postcode"].astype(str)
    first_num_s = recs["addr_numbers"].apply(_first)
    has_key     = postcode_s.str.len().gt(0) & first_num_s.str.len().gt(0)

    addr_key_str = np.where(
        has_key,
        postcode_s + "|" + first_num_s,
        np.where(
            recs["addr_norm"].str.len().gt(0),
            "__anorm__" + recs["addr_norm"],
            "",
        ),
    )
    recs = recs.copy()
    recs["_addr_key_str"] = addr_key_str

    # distinct name_core count per address key (skip empty keys)
    addr_freq = (
        recs[recs["_addr_key_str"] != ""]
        .groupby("_addr_key_str")["name_core"]
        .nunique()
        .rename("addr_freq")
    )
    entity_to_akey = recs.set_index("entity_id")["_addr_key_str"]

    s1_akey   = entity_to_akey.reindex(cands["s1_id"]).values
    cand_akey = entity_to_akey.reindex(cands["cand_id"]).values

    fc_addr_freq_s1 = (
        pd.Series(s1_akey).map(addr_freq).fillna(1).astype(np.float32).values
    )
    fc_addr_freq_cand = (
        pd.Series(cand_akey).map(addr_freq).fillna(1).astype(np.float32).values
    )

    # ------------------------------------------------------------------ #
    # 5.  Per-S1 aggregates: fc_n_cands, fc_n_close, fc_rank, fc_gap_best
    # ------------------------------------------------------------------ #
    n_cands_map = cands.groupby("s1_id")["cand_id"].count().rename("n_cands")

    if "bs_dense" in cands.columns and cands["bs_dense"].notna().any():
        close_mask  = cands["bs_dense"].fillna(0.0) >= 0.95
        n_close_map = (
            cands[close_mask]
            .groupby("s1_id")["cand_id"]
            .count()
            .rename("n_close")
        )
    elif "bs_tfidf_name" in cands.columns and cands["bs_tfidf_name"].notna().any():
        close_mask  = cands["bs_tfidf_name"].fillna(0.0) >= 0.8
        n_close_map = (
            cands[close_mask]
            .groupby("s1_id")["cand_id"]
            .count()
            .rename("n_close")
        )
    else:
        n_close_map = pd.Series(dtype="int64", name="n_close")

    fc_rank = (
        cands["prune_rank"].values.astype(np.float32)
        if "prune_rank" in cands.columns
        else np.zeros(len(cands), dtype=np.float32)
    )

    if "prune_score" in cands.columns:
        best_score_map = cands.groupby("s1_id")["prune_score"].max()
        fc_gap_best = (
            cands["s1_id"].map(best_score_map) - cands["prune_score"]
        ).astype(np.float32).values
    else:
        fc_gap_best = np.zeros(len(cands), dtype=np.float32)

    fc_n_cands = cands["s1_id"].map(n_cands_map).fillna(0).astype(np.float32).values
    fc_n_close = cands["s1_id"].map(n_close_map).fillna(0).astype(np.float32).values

    # ------------------------------------------------------------------ #
    # 6.  Reverse rank: fc_rev_rank, fc_rev_n
    #     For each cand_id: rank this S1 among all S1s that list it
    # ------------------------------------------------------------------ #
    dense_all_path = config.cache_path("emb/dense_neighbors_train_all.parquet")
    if split == "train" and dense_all_path.exists():
        print(f"  [fc_rev] Using {dense_all_path} (full 2.2M train S1) for reverse competition features...")
        dense_all = pd.read_parquet(dense_all_path, columns=["s1_id", "cand_id", "score"])
        rev_n_map = dense_all.groupby("cand_id")["s1_id"].count()
        fc_rev_n = cands["cand_id"].map(rev_n_map).fillna(1).astype(np.float32).values

        dense_all["rev_rank"] = (
            dense_all.groupby("cand_id")["score"]
            .rank(method="first", ascending=False)
            .sub(1)
            .astype(np.float32)
        )
        cands_ranked = cands[["s1_id", "cand_id"]].merge(
            dense_all[["s1_id", "cand_id", "rev_rank"]],
            on=["s1_id", "cand_id"],
            how="left",
        )
        sample_rev_rank = (
            cands.groupby("cand_id")["prune_score"]
            .rank(method="first", ascending=False)
            .sub(1)
            .astype(np.float32)
        ) if "prune_score" in cands.columns else pd.Series(0.0, index=cands.index)
        fc_rev_rank = cands_ranked["rev_rank"].fillna(sample_rev_rank).astype(np.float32).values
    elif "prune_score" in cands.columns:
        # rank within cand_id groups (0 = highest prune_score = best)
        fc_rev_rank = (
            cands.groupby("cand_id")["prune_score"]
            .rank(method="first", ascending=False)
            .sub(1)
            .astype(np.float32)
            .values
        )
        rev_n_map = cands.groupby("cand_id")["s1_id"].count()
        fc_rev_n  = cands["cand_id"].map(rev_n_map).astype(np.float32).values
    else:
        fc_rev_rank = np.zeros(len(cands), dtype=np.float32)
        fc_rev_n    = np.ones(len(cands), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # 7.  Pass-level: fc_n_passes, fc_blk_*, fc_bs_*
    # ------------------------------------------------------------------ #
    blk_cols = [f"blk_{p}" for p in PASSES]
    bs_cols  = [f"bs_{p}"  for p in PASSES]

    fc_n_passes = np.zeros(len(cands), dtype=np.float32)
    for col in blk_cols:
        if col in cands.columns:
            fc_n_passes += cands[col].fillna(False).astype(np.float32).values

    # ------------------------------------------------------------------ #
    # 8.  Assemble output
    # ------------------------------------------------------------------ #
    out = pd.DataFrame({
        "s1_id":   cands["s1_id"].values,
        "cand_id": cands["cand_id"].values,
        # chain / co-location context
        "fc_name_freq_s1":   fc_name_freq_s1,
        "fc_name_freq_cand": fc_name_freq_cand,
        "fc_addr_freq_s1":   fc_addr_freq_s1,
        "fc_addr_freq_cand": fc_addr_freq_cand,
        # per-S1 competition
        "fc_n_cands":  fc_n_cands,
        "fc_n_close":  fc_n_close,
        "fc_rank":     fc_rank,
        "fc_gap_best": fc_gap_best,
        # cross-S1 competition
        "fc_rev_rank": fc_rev_rank,
        "fc_rev_n":    fc_rev_n,
        # pass summary
        "fc_n_passes": fc_n_passes,
        "fc_cand_source": (
            cands["cand_source"].astype(np.float32).values
            if "cand_source" in cands.columns
            else np.full(len(cands), np.nan, dtype=np.float32)
        ),
    })

    # blk_* as 0/1 float32
    for col in blk_cols:
        fc_col = f"fc_{col}"
        if col in cands.columns:
            out[fc_col] = cands[col].fillna(False).astype(np.float32).values
        else:
            out[fc_col] = np.float32(0.0)

    # bs_* pass-through, NaN when pass did not find this pair
    for col in bs_cols:
        fc_col = f"fc_{col}"
        if col in cands.columns:
            out[fc_col] = cands[col].astype(np.float32).values
        else:
            out[fc_col] = np.full(len(cands), np.nan, dtype=np.float32)

    # Guarantee all fc_* are float32
    for c in out.columns:
        if c.startswith("fc_"):
            out[c] = out[c].astype(np.float32)

    # ------------------------------------------------------------------ #
    # 9.  Persist
    # ------------------------------------------------------------------ #
    out_path = config.cache_path(f"feat_ctx_{split}.parquet")
    out.to_parquet(out_path, index=False)
    elapsed = time.perf_counter() - t0
    print(
        f"\n  Saved {len(out):,} rows x {len(out.columns)} columns -> {out_path}"
        f"  ({elapsed:.1f}s)"
    )
    return out


def build_stage2_features(split: str, run_id: str) -> pd.DataFrame:
    """Compute stage-2 context features from stage-1 out-of-fold probabilities.

    Input:
      cache/preds/{run_id}_{split}.parquet  columns: s1_id, cand_id, prob
                                            (+ 'fold' on train - OOF only)
      cache/candidates_{split}.parquet      for cand_source column
      cache/records.parquet                 for name_core (cross-source support)

    Output:
      cache/feat_ctx2_{split}_{run_id}.parquet

    Feature catalogue
    -----------------
    fc2_rank_p            : rank of this prob within the S1 (0 = highest)
    fc2_gap_p             : max prob for this S1 minus this prob
    fc2_second_p          : the S1's second-highest prob (NaN if only 1 cand)
    fc2_rev_rank_p        : rank of this S1 among all S1s listing this cand_id, by prob
    fc2_mutual_best       : 1 if this pair is both the S1's best AND the cand's best S1
    fc2_best_competitor_p : highest prob of this cand_id with any OTHER S1
    fc2_support           : max over S1's other-source candidates a of
                            p(s1, a) * token_set_ratio(name_core(a), name_core(cand)) / 100
    fc2_n_above_05        : number of this S1's candidates with prob > 0.5
    """
    import time as _time
    from collections import defaultdict

    from rapidfuzz.fuzz import token_set_ratio

    t0 = _time.perf_counter()
    print(f"\n=== build_stage2_features(split={split!r}, run_id={run_id!r}) ===")

    # ------------------------------------------------------------------ #
    # 1.  Load predictions
    # ------------------------------------------------------------------ #
    preds_path = config.cache_path(f"preds/{run_id}_{split}.parquet")
    if not preds_path.exists():
        raise FileNotFoundError(
            f"preds/{run_id}_{split}.parquet not found. "
            f"Run train.py / predict.py with run_id={run_id!r} first."
        )
    preds = pd.read_parquet(preds_path, columns=["s1_id", "cand_id", "prob"])
    preds["prob"] = preds["prob"].astype(np.float32)
    print(f"  preds:  {len(preds):,} pairs  (run_id={run_id!r})")

    # ------------------------------------------------------------------ #
    # 2.  Attach cand_source
    # ------------------------------------------------------------------ #
    cands_path = config.cache_path(f"candidates_{split}.parquet")
    if cands_path.exists():
        cands_src = pd.read_parquet(
            cands_path, columns=["s1_id", "cand_id", "cand_source"]
        )
        preds = preds.merge(cands_src, on=["s1_id", "cand_id"], how="left")
    else:
        preds["cand_source"] = preds["cand_id"].str[1].astype(np.int8)
    preds["cand_source"] = preds["cand_source"].fillna(0).astype(np.int8)

    # ------------------------------------------------------------------ #
    # 3.  Load name_core for fc2_support
    # ------------------------------------------------------------------ #
    recs_path = config.cache_path("records.parquet")
    if recs_path.exists():
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(recs_path).names)
        cols = [c for c in ["split", "entity_id", "name_core"] if c in avail]
        recs = pd.read_parquet(recs_path, columns=cols)
        if "split" in recs.columns:
            recs = recs[recs["split"] == split]
    else:
        frames = []
        for src in config.SOURCES:
            raw = io_utils.read_source(split, src)
            raw["name_core"] = raw["business_name"].str.lower().fillna("")
            frames.append(raw[["entity_id", "name_core"]])
        recs = pd.concat(frames, ignore_index=True)
    name_core_map: dict[str, str] = (
        recs[["entity_id", "name_core"]]
        .drop_duplicates("entity_id")
        .set_index("entity_id")["name_core"]
        .to_dict()
    )
    print(f"  records: {len(name_core_map):,} unique entity_ids for support")

    # ------------------------------------------------------------------ #
    # 4.  fc2_rank_p, fc2_gap_p, fc2_second_p, fc2_n_above_05
    # ------------------------------------------------------------------ #
    preds["fc2_rank_p"] = (
        preds.groupby("s1_id")["prob"]
        .rank(method="first", ascending=False)
        .sub(1)
        .astype(np.float32)
    )

    s1_max_p = preds.groupby("s1_id")["prob"].max().rename("_max_p")
    preds = preds.join(s1_max_p, on="s1_id")
    preds["fc2_gap_p"] = (preds["_max_p"] - preds["prob"]).astype(np.float32)
    preds.drop(columns=["_max_p"], inplace=True)

    # S1 second highest prob using sort + cumcount (vectorized, no slow Python groupby.apply)
    s1_sorted = preds[["s1_id", "prob"]].sort_values(["s1_id", "prob"], ascending=[True, False])
    s1_sorted["_rank"] = s1_sorted.groupby("s1_id").cumcount()
    s1_second = s1_sorted[s1_sorted["_rank"] == 1].set_index("s1_id")["prob"].rename("fc2_second_p")
    preds = preds.join(s1_second, on="s1_id")
    preds["fc2_second_p"] = preds["fc2_second_p"].astype(np.float32)

    above_05 = (
        preds[preds["prob"] > 0.5]
        .groupby("s1_id")["cand_id"]
        .count()
        .rename("fc2_n_above_05")
    )
    preds = preds.join(above_05, on="s1_id")
    preds["fc2_n_above_05"] = preds["fc2_n_above_05"].fillna(0).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 5.  fc2_rev_rank_p, fc2_mutual_best, fc2_best_competitor_p
    # ------------------------------------------------------------------ #
    preds["fc2_rev_rank_p"] = (
        preds.groupby("cand_id")["prob"]
        .rank(method="first", ascending=False)
        .sub(1)
        .astype(np.float32)
    )

    preds["fc2_mutual_best"] = (
        (preds["fc2_rank_p"] == 0) & (preds["fc2_rev_rank_p"] == 0)
    ).astype(np.float32)

    cand_max_p = preds.groupby("cand_id")["prob"].max().rename("_cand_max_p")
    cand_sorted = preds[["cand_id", "prob"]].sort_values(["cand_id", "prob"], ascending=[True, False])
    cand_sorted["_rank"] = cand_sorted.groupby("cand_id").cumcount()
    cand_2nd_p = cand_sorted[cand_sorted["_rank"] == 1].set_index("cand_id")["prob"].rename("_cand_2nd_p")
    preds = preds.join(cand_max_p, on="cand_id").join(cand_2nd_p, on="cand_id")

    # If this row is the cand's max, its best competitor is the second-max;
    # otherwise it's the overall max. Use >= to handle float ties correctly.
    is_cand_max = preds["prob"] >= preds["_cand_max_p"]
    preds["fc2_best_competitor_p"] = np.where(
        is_cand_max,
        preds["_cand_2nd_p"],
        preds["_cand_max_p"],
    ).astype(np.float32)
    preds.drop(columns=["_cand_max_p", "_cand_2nd_p"], inplace=True)

    # ------------------------------------------------------------------ #
    # 6.  fc2_support (vectorized cross-source support using cdist)
    #
    #   For candidate c (source X) of S1 e:
    #     max over e's candidates a from OTHER source (top 5 by prob) of
    #       p(e, a) * token_set_ratio(name_core_a, name_core_c) / 100
    # ------------------------------------------------------------------ #
    print("  Computing fc2_support (vectorized) ...", end="", flush=True)

    preds["_src_rank"] = (
        preds.groupby(["s1_id", "cand_source"])["prob"]
        .rank(method="first", ascending=False)
    )
    preds_top5 = preds[preds["_src_rank"] <= 5].copy()

    has_s2 = set(preds_top5.loc[preds_top5["cand_source"] == 2, "s1_id"].unique())
    has_s3 = set(preds_top5.loc[preds_top5["cand_source"] == 3, "s1_id"].unique())
    s1_both = has_s2 & has_s3

    support_map: dict[tuple[str, str], float] = {}

    if s1_both:
        both_df = preds_top5[preds_top5["s1_id"].isin(s1_both)].copy()
        both_df["name"] = both_df["cand_id"].map(name_core_map).fillna("").astype(str)

        try:
            from rapidfuzz.process import cdist
            use_cdist = True
        except ImportError:
            use_cdist = False

        for s1_id, grp in both_df.groupby("s1_id"):
            g2 = grp[grp["cand_source"] == 2]
            g3 = grp[grp["cand_source"] == 3]
            if len(g2) == 0 or len(g3) == 0:
                continue

            names2 = g2["name"].tolist()
            names3 = g3["name"].tolist()
            cands2 = g2["cand_id"].tolist()
            cands3 = g3["cand_id"].tolist()
            probs2 = g2["prob"].values
            probs3 = g3["prob"].values

            if use_cdist:
                sim_mat = cdist(names2, names3, scorer=token_set_ratio, dtype=np.float32) / 100.0
            else:
                sim_mat = np.array([
                    [token_set_ratio(n2, n3) / 100.0 for n3 in names3]
                    for n2 in names2
                ], dtype=np.float32)

            sup2 = (sim_mat * probs3[np.newaxis, :]).max(axis=1)
            for cid, val in zip(cands2, sup2):
                support_map[(s1_id, cid)] = float(val)

            sup3 = (sim_mat * probs2[:, np.newaxis]).max(axis=0)
            for cid, val in zip(cands3, sup3):
                support_map[(s1_id, cid)] = float(val)

    pair_series = pd.Series(list(zip(preds["s1_id"], preds["cand_id"])))
    preds["fc2_support"] = pair_series.map(support_map).fillna(0.0).astype(np.float32).values
    preds.drop(columns=["_src_rank"], inplace=True)
    print(" done")

    # ------------------------------------------------------------------ #
    # 7.  Assemble and persist
    # ------------------------------------------------------------------ #
    fc2_cols = [
        "fc2_rank_p", "fc2_gap_p", "fc2_second_p",
        "fc2_rev_rank_p", "fc2_mutual_best", "fc2_best_competitor_p",
        "fc2_support", "fc2_n_above_05",
    ]
    out = preds[["s1_id", "cand_id"] + fc2_cols].copy()
    for c in fc2_cols:
        out[c] = out[c].astype(np.float32)

    out_path = config.cache_path(f"feat_ctx2_{split}_{run_id}.parquet")
    out.to_parquet(out_path, index=False)
    elapsed = _time.perf_counter() - t0
    print(
        f"\n  Saved {len(out):,} rows x {len(out.columns)} columns -> {out_path}"
        f"  ({elapsed:.1f}s)"
    )
    return out


# ---------------------------------------------------------------------------
# CLI  (diagnostic describe() only in __main__)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build context features for one split."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--stage1-run",
        help="run_id of stage-1 predictions; builds fc2_* features when given",
    )
    args = parser.parse_args()

    if args.stage1_run:
        feat = build_stage2_features(args.split, args.stage1_run)
        fc2_cols = [c for c in feat.columns if c.startswith("fc2_")]

        # Diagnostic: describe() by label (GT accessed here only)
        if args.split == "train":
            try:
                gt_raw = io_utils.read_ground_truth()
                gt_pairs: set[tuple[str, str]] = set()
                for row in gt_raw.itertuples(index=False):
                    for m in row.matches:
                        gt_pairs.add((row.s1_id, m))
                feat["_label"] = [
                    1 if (s, c) in gt_pairs else 0
                    for s, c in zip(feat["s1_id"], feat["cand_id"])
                ]
                print("\n--- describe() by label (fc2_*) ---")
                for lbl, grp in feat.groupby("_label"):
                    lbl_str = "POSITIVE" if lbl == 1 else "negative"
                    print(f"\n  label={lbl_str}  (n={len(grp):,})")
                    print(grp[fc2_cols].describe().to_string())
            except Exception as exc:
                print(f"\n  [diagnostic skipped: {exc}]")

    else:
        feat = build_context_features(args.split)
        fc_cols = [c for c in feat.columns if c.startswith("fc_")]

        # Diagnostic: describe() by label (GT accessed here only)
        if args.split == "train":
            try:
                gt_raw = io_utils.read_ground_truth()
                gt_pairs_fc: set[tuple[str, str]] = set()
                for row in gt_raw.itertuples(index=False):
                    for m in row.matches:
                        gt_pairs_fc.add((row.s1_id, m))

                feat["_label"] = [
                    1 if (s, c) in gt_pairs_fc else 0
                    for s, c in zip(feat["s1_id"], feat["cand_id"])
                ]
                print("\n--- describe() by label (fc_*) ---")
                for lbl, grp in feat.groupby("_label"):
                    lbl_str = "POSITIVE" if lbl == 1 else "negative"
                    print(f"\n  label={lbl_str}  (n={len(grp):,})")
                    print(grp[fc_cols].describe().to_string())
            except Exception as exc:
                print(f"\n  [diagnostic skipped: {exc}]")


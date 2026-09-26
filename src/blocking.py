"""Candidate generation (blocking). Owner: Siva (SI-1, SI-3, SI-6).

Input:  cache/records.parquet, plus cache/emb/dense_neighbors_{split}.parquet from Adithya for the dense pass
Output: cache/candidates_{split}.parquet, the FINAL pruned candidate set (schema: docs/INTERFACES.md).
That file is exactly what the model scores and exactly what goes into candidate_pairs.tsv.

Each pass runs within every country value found in the data (an open set: never list countries by hand),
separately for S1->S2 and S1->S3. Target: >= 99% pair recall at roughly 30-50 candidates per S1.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from . import io_utils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASSES = ("tfidf_name", "tfidf_full", "rare_token", "addr_key", "dense", "reverse")

# K nearest neighbours to retrieve per S1 per target source per country
TFIDF_K = 20

# Number of S1 rows to process per cosine-similarity chunk (keeps RAM bounded)
CHUNK_SIZE = 2_000


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_records(split: str) -> pd.DataFrame:
    """Load cache/records.parquet for *split*; fall back to raw TSV if absent.

    Returns columns: split, source (int8), entity_id, country, name_core.
    """
    recs_path = config.cache_path("records.parquet")
    if recs_path.exists():
        df = pd.read_parquet(
            recs_path,
            columns=["split", "source", "entity_id", "country", "name_core"],
        )
        df = df[df["split"] == split].copy()
        df["name_core"] = df["name_core"].fillna("").astype(str)
    else:
        # Fallback: build name_core from lowercased business_name via io_utils
        frames = []
        for src in config.SOURCES:
            raw = io_utils.read_source(split, src)
            raw["source"] = np.int8(src)
            raw["split"] = split
            raw["name_core"] = raw["business_name"].str.lower().fillna("")
            raw["country"] = raw["country"].fillna("").str.strip()
            frames.append(raw[["split", "source", "entity_id", "country", "name_core"]])
        df = pd.concat(frames, ignore_index=True)

    df["source"] = df["source"].astype(np.int8)
    return df


def _top_k_per_row_chunk(
    s1_mat: csr_matrix,
    pool_mat: csr_matrix,
    k: int,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (global_s1_row_indices, pool_col_indices, scores) for top-k hits.

    Processes CHUNK_SIZE S1 rows at a time; never materialises a dense
    S1 x pool matrix for the whole query set.
    """
    n_s1 = s1_mat.shape[0]
    actual_k = min(k, pool_mat.shape[0])
    if actual_k == 0:
        return

    for start in range(0, n_s1, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n_s1)
        chunk = s1_mat[start:end]                   # still sparse

        # Sparse dot product -> (chunk_size x pool_size) dense chunk only
        sim_arr = (chunk @ pool_mat.T).toarray()    # float64 from sklearn norms

        # argpartition: top-k columns per row (unordered, sufficient for recall)
        part = np.argpartition(sim_arr, -actual_k, axis=1)[:, -actual_k:]

        local_rows = np.repeat(np.arange(end - start), actual_k)
        col_idxs = part.ravel()
        scores = sim_arr[local_rows, col_idxs]

        mask = scores > 0.0
        if mask.any():
            yield (
                local_rows[mask] + start,   # global S1 index
                col_idxs[mask],
                scores[mask].astype(np.float32),
            )


# ---------------------------------------------------------------------------
# Pass: tfidf_name (SI-1)
# ---------------------------------------------------------------------------

def _run_tfidf_name(split: str, records: pd.DataFrame) -> pd.DataFrame:
    """TF-IDF character n-gram blocking on name_core.

    For every country present in S1 of this split, and for each of target
    sources 2 and 3 separately:
      - Fit TfidfVectorizer on ALL records of the split (all sources, all countries).
      - Transform S1 (country-filtered, optionally subsampled) and target pool.
      - Retrieve top TFIDF_K cosine neighbours per S1 row via chunked sparse product.
      - Falls back to whole target-source pool when no country match exists.

    Returns DataFrame with columns [s1_id, cand_id, cand_source, blk_tfidf_name, bs_tfidf_name].
    """
    t0 = time.perf_counter()

    # ---- Determine active S1 rows (apply dev-sample if needed) ----
    s1_all = records[records["source"] == 1].copy()
    if config.SAMPLE_FRAC < 1.0:
        s1_mask = s1_all["entity_id"].map(io_utils.in_dev_sample)
        s1_active = s1_all[s1_mask].copy()
    else:
        s1_active = s1_all.copy()

    countries = s1_active["country"].unique()

    # ---- Fit ONE vectoriser on the whole split corpus ----
    vect = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    vect.fit(records["name_core"].values)
    print(
        f"  [tfidf_name] vectoriser fitted: vocab={len(vect.vocabulary_):,}  "
        f"countries={len(countries)}  active_s1={len(s1_active):,}"
    )

    all_pairs: list[pd.DataFrame] = []

    for country in countries:
        s1_cty = s1_active[s1_active["country"] == country]
        if len(s1_cty) == 0:
            continue

        s1_mat = vect.transform(s1_cty["name_core"].values)
        s1_ids = s1_cty["entity_id"].values

        for tgt_src in (2, 3):
            # Pool: target-source records for this country
            pool_cty = records[
                (records["source"] == tgt_src) & (records["country"] == country)
            ]
            fallback = False
            if len(pool_cty) == 0:
                # No target records for this country: fall back to whole source pool
                pool_cty = records[records["source"] == tgt_src]
                fallback = True
            if len(pool_cty) == 0:
                continue

            pool_mat = vect.transform(pool_cty["name_core"].values)
            pool_ids = pool_cty["entity_id"].values

            rows_list, cols_list, scores_list = [], [], []
            for row_idxs, col_idxs, sc in _top_k_per_row_chunk(s1_mat, pool_mat, TFIDF_K):
                rows_list.append(row_idxs)
                cols_list.append(col_idxs)
                scores_list.append(sc)

            if not rows_list:
                continue

            row_arr = np.concatenate(rows_list)
            col_arr = np.concatenate(cols_list)
            sc_arr = np.concatenate(scores_list).astype(np.float32)

            pairs = pd.DataFrame({
                "s1_id":      s1_ids[row_arr],
                "cand_id":    pool_ids[col_arr],
                "cand_source": np.int8(tgt_src),
                "bs_tfidf_name": sc_arr,
            })
            all_pairs.append(pairs)

            n_s1_cty = len(s1_cty)
            mean_c = len(pairs) / n_s1_cty if n_s1_cty else 0
            print(
                f"    {country:30s}  src->{tgt_src}  "
                f"s1={n_s1_cty:>7,}  pool={len(pool_cty):>8,}{'(fallback)' if fallback else '':10s}  "
                f"pairs={len(pairs):>8,}  mean/s1={mean_c:.1f}"
            )

    elapsed = time.perf_counter() - t0

    if not all_pairs:
        result = pd.DataFrame(
            columns=["s1_id", "cand_id", "cand_source", "bs_tfidf_name"]
        )
    else:
        result = (
            pd.concat(all_pairs, ignore_index=True)
            .drop_duplicates(subset=["s1_id", "cand_id"])
            .reset_index(drop=True)
        )

    # ---- Summary stats ----
    print(f"\n  [tfidf_name][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    if len(result):
        cand_counts = result.groupby("s1_id")["cand_id"].count()
        print(
            f"  candidates/S1:  mean={cand_counts.mean():.1f}  "
            f"p50={cand_counts.median():.0f}  p95={cand_counts.quantile(0.95):.0f}"
        )

    result["blk_tfidf_name"] = True
    return result


# ---------------------------------------------------------------------------
# Main entry-point: generate_candidates
# ---------------------------------------------------------------------------

def generate_candidates(split: str) -> pd.DataFrame:
    """Union of all passes, with INTERFACES columns, saved to cache/candidates_{split}.parquet.

    SI-1: implements the 'tfidf_name' pass only.
    All other passes' blk_* columns are set to False and bs_* to NaN,
    ready to be filled in by future tasks (SI-3, SI-6).
    """
    t_total = time.perf_counter()
    print(f"\n=== generate_candidates(split={split!r}) ===")

    records = _load_records(split)
    src_counts = records["source"].value_counts().sort_index().to_dict()
    print(f"  Loaded {len(records):,} records  {src_counts}")

    # ---- Run tfidf_name pass ----
    cands = _run_tfidf_name(split, records)

    # ---- Stub columns for passes not yet implemented ----
    for pass_name in PASSES:
        blk_col = f"blk_{pass_name}"
        bs_col  = f"bs_{pass_name}"
        if blk_col not in cands.columns:
            cands[blk_col] = False
        if bs_col not in cands.columns:
            cands[bs_col] = np.float32(np.nan)

    # Enforce types
    for pass_name in PASSES:
        cands[f"blk_{pass_name}"] = cands[f"blk_{pass_name}"].astype(bool)
        cands[f"bs_{pass_name}"]  = cands[f"bs_{pass_name}"].astype(np.float32)

    # ---- prune_score: max score across all implemented passes ----
    bs_cols = [f"bs_{p}" for p in PASSES]
    cands["prune_score"] = cands[bs_cols].max(axis=1).astype(np.float32)

    # ---- prune_rank: 0 = best match within each S1, descending ----
    cands["prune_rank"] = (
        cands
        .groupby("s1_id")["prune_score"]
        .rank(method="first", ascending=False)
        .sub(1)
        .astype(np.int16)
    )

    # ---- Column order per INTERFACES.md ----
    col_order = (
        ["s1_id", "cand_id", "cand_source"]
        + [f"blk_{p}" for p in PASSES]
        + [f"bs_{p}"  for p in PASSES]
        + ["prune_score", "prune_rank"]
    )
    cands = cands[col_order]

    # ---- Persist ----
    out_path = config.cache_path(f"candidates_{split}.parquet")
    cands.to_parquet(out_path, index=False)
    print(
        f"\n  Saved {len(cands):,} candidate pairs → {out_path}"
        f"  (total elapsed: {time.perf_counter() - t_total:.1f}s)"
    )
    return cands


# ---------------------------------------------------------------------------
# SI-2: Blocking diagnostics
# ---------------------------------------------------------------------------

def _f05_entity(pred: set[str], true: set[str]) -> float:
    """F0.5 for one S1 entity (mirrors metrics.f05_entity; swap when AR-1 lands)."""
    if not pred and not true:
        return 1.0
    if not pred or not true:
        return 0.0
    tp = len(pred & true)
    if tp == 0:
        return 0.0
    p  = tp / len(pred)
    r  = tp / len(true)
    return 1.25 * p * r / (0.25 * p + r)


def _blocking_report(
    candidates: pd.DataFrame,
    truth: dict[str, list[str]],
) -> dict:
    """Compute blocking diagnostics.

    Mirrors the signature of metrics.blocking_report (AR-1). Swap in when ready.

    Parameters
    ----------
    candidates : DataFrame with at least columns [s1_id, cand_id] plus
                 blk_{pass} bool columns for every pass in PASSES.
    truth      : {s1_id -> [cand_id, ...]}  (from io_utils.read_ground_truth)

    Returns
    -------
    dict with keys:
        pair_recall       – found GT pairs / all GT pairs
        entity_ceiling    – share of S1 whose *all* true matches are in candidates
        oracle_f05        – macro F0.5 when predicting exactly the true matches inside candidates
        mean_cands        – mean candidates per S1
        p50_cands         – median candidates per S1
        p95_cands         – 95th-percentile candidates per S1
        total_pairs       – total candidate pairs
        pass_recall       – {pass_name: recall} (union of that pass)
        marginal_recall   – {pass_name: recall drop if that pass is removed}
        country_stats     – list-of-dicts per country
        source_stats      – {2: dict, 3: dict}
    """
    # Index candidates for fast lookup
    cands_set = (
        candidates
        .groupby("s1_id")["cand_id"]
        .apply(set)
        .to_dict()
    )

    # Ground-truth pairs as a flat set of (s1_id, cand_id)
    gt_pairs: set[tuple[str, str]] = set()
    for s1, matches in truth.items():
        for m in matches:
            gt_pairs.add((s1, m))

    # Build (s1_id, cand_id) candidate pair set
    cand_pairs: set[tuple[str, str]] = set(
        zip(candidates["s1_id"], candidates["cand_id"])
    )

    # --- Overall pair recall ---
    found      = gt_pairs & cand_pairs
    pair_recall = len(found) / len(gt_pairs) if gt_pairs else 1.0

    # --- Entity-level ceiling recall ---
    # An S1 is "covered" if ALL its true matches are in the candidates
    n_covered = 0
    n_with_matches = 0
    oracle_scores: list[float] = []

    for s1, true_matches in truth.items():
        if not true_matches:
            # singleton: oracle predicts nothing -> F0.5 = 1.0
            oracle_scores.append(1.0)
            continue
        n_with_matches += 1
        pool = cands_set.get(s1, set())
        recoverable = set(true_matches) & pool
        if len(recoverable) == len(true_matches):
            n_covered += 1
        oracle_scores.append(_f05_entity(recoverable, set(true_matches)))

    entity_ceiling = n_covered / n_with_matches if n_with_matches else 1.0
    oracle_f05     = float(np.mean(oracle_scores)) if oracle_scores else 1.0

    # --- Candidates/S1 distribution ---
    counts = candidates.groupby("s1_id")["cand_id"].count()
    mean_cands = float(counts.mean())
    p50_cands  = float(counts.quantile(0.50))
    p95_cands  = float(counts.quantile(0.95))
    total_pairs = int(len(candidates))

    # --- Per-pass recall (union within each pass) ---
    pass_recall: dict[str, float] = {}
    for p in PASSES:
        blk_col = f"blk_{p}"
        if blk_col not in candidates.columns:
            pass_recall[p] = 0.0
            continue
        pass_pairs = set(
            zip(
                candidates.loc[candidates[blk_col], "s1_id"],
                candidates.loc[candidates[blk_col], "cand_id"],
            )
        )
        pass_recall[p] = len(gt_pairs & pass_pairs) / len(gt_pairs) if gt_pairs else 1.0

    # --- Marginal recall: recall drop when a pass is removed ---
    marginal_recall: dict[str, float] = {}
    for p in PASSES:
        blk_col = f"blk_{p}"
        if blk_col not in candidates.columns or not candidates[blk_col].any():
            marginal_recall[p] = 0.0
            continue
        # Pairs covered by at least one OTHER pass
        other_blk = [f"blk_{q}" for q in PASSES if q != p and f"blk_{q}" in candidates.columns]
        if other_blk:
            covered_by_others = candidates[other_blk].any(axis=1)
            other_pairs = set(
                zip(
                    candidates.loc[covered_by_others, "s1_id"],
                    candidates.loc[covered_by_others, "cand_id"],
                )
            )
            recall_without = len(gt_pairs & other_pairs) / len(gt_pairs) if gt_pairs else 1.0
        else:
            recall_without = 0.0
        marginal_recall[p] = pair_recall - recall_without

    # --- Per-country stats ---
    country_stats: list[dict] = []
    if "country" not in candidates.columns:
        # Pull country from records for join
        recs_path = config.cache_path("records.parquet")
        if recs_path.exists():
            s1_cty = pd.read_parquet(
                recs_path,
                columns=["split", "source", "entity_id", "country"],
                filters=[("split", "==", "train"), ("source", "==", 1)],
            ).rename(columns={"entity_id": "s1_id"})
        else:
            raw = io_utils.read_source("train", 1)
            s1_cty = raw.rename(columns={"entity_id": "s1_id"})[["s1_id", "country"]]
        cands_with_cty = candidates.merge(s1_cty[["s1_id", "country"]], on="s1_id", how="left")
    else:
        cands_with_cty = candidates

    # GT per country: build a map s1 -> country from the candidates
    s1_to_country = (
        cands_with_cty.drop_duplicates("s1_id")
        .set_index("s1_id")["country"]
        .to_dict()
    )

    for country, grp in cands_with_cty.groupby("country", sort=True):
        grp_s1_ids = set(grp["s1_id"].unique())
        gt_cty = {s: m for s, m in truth.items() if s1_to_country.get(s) == country}
        gt_pairs_cty: set[tuple[str, str]] = set()
        for s1, matches in gt_cty.items():
            for m in matches:
                gt_pairs_cty.add((s1, m))
        cand_pairs_cty = set(zip(grp["s1_id"], grp["cand_id"]))
        recall_cty = len(gt_pairs_cty & cand_pairs_cty) / len(gt_pairs_cty) if gt_pairs_cty else 1.0
        cc = grp.groupby("s1_id")["cand_id"].count()
        country_stats.append({
            "country":    country,
            "s1":         len(grp_s1_ids),
            "gt_pairs":   len(gt_pairs_cty),
            "recall":     recall_cty,
            "mean_cands": float(cc.mean()),
            "p50_cands":  float(cc.quantile(0.50)),
            "p95_cands":  float(cc.quantile(0.95)),
        })

    # --- Per-target-source stats ---
    source_stats: dict[int, dict] = {}
    for tgt_src in (2, 3):
        src_col = "cand_source"
        if src_col in candidates.columns:
            grp_src = candidates[candidates[src_col] == tgt_src]
        else:
            grp_src = candidates
        gt_src = {(s, m) for s, m in gt_pairs if m.startswith(f"S{tgt_src}-")}
        cand_src = set(zip(grp_src["s1_id"], grp_src["cand_id"]))
        recall_src = len(gt_src & cand_src) / len(gt_src) if gt_src else 1.0
        source_stats[tgt_src] = {"gt_pairs": len(gt_src), "recall": recall_src, "pairs": len(grp_src)}

    return {
        "pair_recall":    pair_recall,
        "entity_ceiling": entity_ceiling,
        "oracle_f05":     oracle_f05,
        "mean_cands":     mean_cands,
        "p50_cands":      p50_cands,
        "p95_cands":      p95_cands,
        "total_pairs":    total_pairs,
        "pass_recall":    pass_recall,
        "marginal_recall": marginal_recall,
        "country_stats":  country_stats,
        "source_stats":   source_stats,
    }


def report(split: str = "train") -> dict:
    """Load cache/candidates_{split}.parquet, compute and print blocking diagnostics.

    Only meaningful for split='train' (ground truth available).  For 'test'
    the recall numbers are not computable; a warning is printed instead.

    Appends one row to docs/blocking_log.csv.
    Returns the metrics dict from _blocking_report.
    """
    if split != "train":
        print(f"[report] No ground truth for split={split!r}; skipping recall computation.")
        return {}

    cands_path = config.cache_path(f"candidates_{split}.parquet")
    if not cands_path.exists():
        raise FileNotFoundError(
            f"candidates_{split}.parquet not found. Run generate_candidates('{split}') first."
        )

    cands = pd.read_parquet(cands_path)
    print(f"\n=== blocking report (split={split!r}) ===")
    print(f"  Loaded {len(cands):,} candidate pairs from {cands_path}")

    # Load ground truth
    gt_raw = io_utils.read_ground_truth()   # DataFrame: s1_id, matches (list[str])
    truth  = dict(zip(gt_raw["s1_id"], gt_raw["matches"]))
    print(f"  Ground-truth: {len(truth):,} S1 entities, "
          f"{sum(len(v) for v in truth.values()):,} true pairs")

    # Try using metrics.blocking_report if AR-1 has landed; fall back to local
    try:
        from . import metrics as _m
        stats = _m.blocking_report(cands, truth)
    except (NotImplementedError, AttributeError):
        stats = _blocking_report(cands, truth)

    # ----------------------------------------------------------------
    # Print table
    # ----------------------------------------------------------------
    sep  = "-" * 72
    sep2 = "=" * 72

    print(f"\n{sep2}")
    print(f"  OVERALL")
    print(f"{sep2}")
    print(f"  pair recall        : {stats['pair_recall']:.4f}  "
          f"({stats['pair_recall']*100:.2f}%)")
    print(f"  entity ceiling     : {stats['entity_ceiling']:.4f}  "
          f"(share of S1 with ALL true matches found)")
    print(f"  oracle F0.5        : {stats['oracle_f05']:.4f}")
    print(f"  total pairs        : {stats['total_pairs']:,}")
    print(f"  candidates/S1      : mean={stats['mean_cands']:.1f}  "
          f"p50={stats['p50_cands']:.0f}  p95={stats['p95_cands']:.0f}")

    # Per-target-source
    print(f"\n{sep}")
    print(f"  {'Source':<10} {'GT pairs':>10} {'Recall':>8} {'Cand pairs':>12}")
    print(sep)
    for src, s in stats.get("source_stats", {}).items():
        print(f"  S{src:<9} {s['gt_pairs']:>10,} {s['recall']:>8.4f} {s['pairs']:>12,}")

    # Per-pass recall & marginal
    print(f"\n{sep}")
    print(f"  {'Pass':<16} {'Recall':>8} {'Marginal':>10}")
    print(sep)
    for p in PASSES:
        r = stats["pass_recall"].get(p, 0.0)
        m = stats["marginal_recall"].get(p, 0.0)
        marker = " <-- active" if r > 0 else ""
        print(f"  {p:<16} {r:>8.4f} {m:>10.4f}{marker}")

    # Per-country breakdown
    print(f"\n{sep}")
    hdr = f"  {'Country':<30} {'S1':>7} {'GT pairs':>9} {'Recall':>8} {'mean':>7} {'p50':>6} {'p95':>6}"
    print(hdr)
    print(sep)
    for cs in sorted(stats.get("country_stats", []), key=lambda x: -x["gt_pairs"]):
        print(
            f"  {cs['country']:<30} {cs['s1']:>7,} {cs['gt_pairs']:>9,} "
            f"{cs['recall']:>8.4f} {cs['mean_cands']:>7.1f} "
            f"{cs['p50_cands']:>6.0f} {cs['p95_cands']:>6.0f}"
        )
    print(sep2)

    # ----------------------------------------------------------------
    # Append to docs/blocking_log.csv
    # ----------------------------------------------------------------
    log_path = config.ROOT / "docs" / "blocking_log.csv"
    fieldnames = [
        "date_ist", "split", "passes", "tfidf_k", "chunk_size",
        "total_pairs", "pair_recall", "entity_ceiling", "oracle_f05",
        "mean_cands", "p50_cands", "p95_cands",
    ]
    row = {
        "date_ist":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "split":          split,
        "passes":         "|".join(p for p in PASSES if stats["pass_recall"].get(p, 0) > 0),
        "tfidf_k":        TFIDF_K,
        "chunk_size":     CHUNK_SIZE,
        "total_pairs":    stats["total_pairs"],
        "pair_recall":    f"{stats['pair_recall']:.6f}",
        "entity_ceiling": f"{stats['entity_ceiling']:.6f}",
        "oracle_f05":     f"{stats['oracle_f05']:.6f}",
        "mean_cands":     f"{stats['mean_cands']:.2f}",
        "p50_cands":      f"{stats['p50_cands']:.0f}",
        "p95_cands":      f"{stats['p95_cands']:.0f}",
    }
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\n  Appended row to {log_path}")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blocking candidates for one split.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--report", action="store_true",
                        help="After generating, run the blocking report (train only).")
    args = parser.parse_args()
    generate_candidates(args.split)
    if args.report:
        report(args.split)


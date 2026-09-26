"""Candidate generation (blocking). Owner: Siva (SI-1, SI-3, SI-6).

Input:  cache/records.parquet, plus cache/emb/dense_neighbors_{split}.parquet from Adithya for the dense pass
Output: cache/candidates_{split}.parquet, the FINAL pruned candidate set (schema: docs/INTERFACES.md).
That file is exactly what the model scores and exactly what goes into candidate_pairs.tsv.

Each pass runs within every country value found in the data (an open set: never list countries by hand),
separately for S1->S2 and S1->S3. Target: >= 99% pair recall at roughly 30-50 candidates per S1.
"""
from __future__ import annotations

import argparse
import time
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
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate blocking candidates for one split.")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()
    generate_candidates(args.split)


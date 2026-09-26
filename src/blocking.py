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


# Rare-token inverted-index: a name_core token is "rare" if it appears in at
# most this many documents across the whole split (all sources, all countries).
RARE_DF_THRESH = 20

# addr_key and rare_token caps (per S1, per target source, per country)
ADDR_KEY_CAP  = 30
RARE_TOKEN_CAP = 30

# reverse pass: how many top S1 to pull per target record
REVERSE_K = 5


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_records(split: str) -> pd.DataFrame:
    """Load cache/records.parquet for *split* with name_core only; fall back to raw TSV.

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


def _load_records_full(split: str) -> pd.DataFrame:
    """Like _load_records but also loads addr_norm, addr_numbers, postcode, city.

    Falls back gracefully: missing columns are filled with empty strings / lists.
    """
    recs_path = config.cache_path("records.parquet")
    want = ["split", "source", "entity_id", "country",
            "name_core", "addr_norm", "addr_numbers", "postcode", "city"]
    if recs_path.exists():
        import pyarrow.parquet as pq
        schema_cols = set(pq.read_schema(recs_path).names)
        cols = [c for c in want if c in schema_cols]
        df = pd.read_parquet(recs_path, columns=cols)
        df = df[df["split"] == split].copy()
    else:
        df = _load_records(split)
        cols = list(df.columns)

    # Ensure all expected columns exist with safe defaults
    for col in ["addr_norm", "postcode", "city"]:
        if col not in df.columns:
            df[col] = ""
        else:
            df[col] = df[col].fillna("").astype(str)
    if "addr_numbers" not in df.columns:
        df["addr_numbers"] = [[] for _ in range(len(df))]
    else:
        df["addr_numbers"] = df["addr_numbers"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    if "name_core" not in df.columns:
        df["name_core"] = ""
    else:
        df["name_core"] = df["name_core"].fillna("").astype(str)

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


def _run_tfidf_name_with_vect(
    split: str,
    records: pd.DataFrame,
    s1_active: pd.DataFrame,
    vect: TfidfVectorizer,
) -> pd.DataFrame:
    """tfidf_name pass re-using a pre-fitted vectoriser.

    Called by generate_candidates() which fits ONE shared vectoriser for
    both tfidf_name and reverse.  Same country/fallback logic as _run_tfidf_name.
    Returns DataFrame[s1_id, cand_id, cand_source, blk_tfidf_name, bs_tfidf_name].
    """
    t0 = time.perf_counter()
    countries = s1_active["country"].unique()
    all_pairs: list[pd.DataFrame] = []

    for country in countries:
        s1_cty = s1_active[s1_active["country"] == country]
        if len(s1_cty) == 0:
            continue
        s1_mat = vect.transform(s1_cty["name_core"].values)
        s1_ids = s1_cty["entity_id"].values

        for tgt_src in (2, 3):
            pool_cty = records[
                (records["source"] == tgt_src) & (records["country"] == country)
            ]
            fallback = False
            if len(pool_cty) == 0:
                pool_cty = records[records["source"] == tgt_src]
                fallback = True
            if len(pool_cty) == 0:
                continue
            pool_mat = vect.transform(pool_cty["name_core"].values)
            pool_ids = pool_cty["entity_id"].values

            rows_l, cols_l, sc_l = [], [], []
            for ri, ci, sc in _top_k_per_row_chunk(s1_mat, pool_mat, TFIDF_K):
                rows_l.append(ri); cols_l.append(ci); sc_l.append(sc)
            if not rows_l:
                continue
            row_arr = np.concatenate(rows_l)
            col_arr = np.concatenate(cols_l)
            sc_arr  = np.concatenate(sc_l).astype(np.float32)
            pairs = pd.DataFrame({
                "s1_id":         s1_ids[row_arr],
                "cand_id":       pool_ids[col_arr],
                "cand_source":   np.int8(tgt_src),
                "bs_tfidf_name": sc_arr,
            })
            all_pairs.append(pairs)
            mean_c = len(pairs) / len(s1_cty)
            print(
                f"    {country:30s}  src->{tgt_src}  "
                f"s1={len(s1_cty):>7,}  pool={len(pool_cty):>8,}"
                f"{'(fallback)' if fallback else '':10s}  "
                f"pairs={len(pairs):>8,}  mean/s1={mean_c:.1f}"
            )

    elapsed = time.perf_counter() - t0
    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_tfidf_name"])
    )
    print(f"\n  [tfidf_name][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    result["blk_tfidf_name"] = True
    return result


# ---------------------------------------------------------------------------
# Pass: tfidf_full (SI-3)
# ---------------------------------------------------------------------------

def _run_tfidf_full(
    split: str,
    records: pd.DataFrame,       # must include addr_norm
    s1_active: pd.DataFrame,
    vect_name: TfidfVectorizer,  # reuse the name vectoriser for fallback country logic
) -> pd.DataFrame:
    """TF-IDF on name_core + ' ' + addr_norm; K=20.

    Same country loop / fallback / sample logic as tfidf_name.
    Returns DataFrame[s1_id, cand_id, cand_source, bs_tfidf_full].
    """
    t0 = time.perf_counter()
    # Build combined text for the whole split corpus
    full_text = (records["name_core"] + " " + records["addr_norm"]).values

    vect = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        sublinear_tf=True,
    )
    vect.fit(full_text)
    print(
        f"  [tfidf_full] vocab={len(vect.vocabulary_):,}  active_s1={len(s1_active):,}"
    )

    countries = s1_active["country"].unique()
    all_pairs: list[pd.DataFrame] = []

    for country in countries:
        s1_cty = s1_active[s1_active["country"] == country]
        if len(s1_cty) == 0:
            continue
        s1_text = (s1_cty["name_core"] + " " + s1_cty["addr_norm"]).values
        s1_mat  = vect.transform(s1_text)
        s1_ids  = s1_cty["entity_id"].values

        for tgt_src in (2, 3):
            pool_cty = records[
                (records["source"] == tgt_src) & (records["country"] == country)
            ]
            fallback = False
            if len(pool_cty) == 0:
                pool_cty = records[records["source"] == tgt_src]
                fallback = True
            if len(pool_cty) == 0:
                continue
            pool_text = (pool_cty["name_core"] + " " + pool_cty["addr_norm"]).values
            pool_mat  = vect.transform(pool_text)
            pool_ids  = pool_cty["entity_id"].values

            rows_l, cols_l, sc_l = [], [], []
            for ri, ci, sc in _top_k_per_row_chunk(s1_mat, pool_mat, TFIDF_K):
                rows_l.append(ri); cols_l.append(ci); sc_l.append(sc)
            if not rows_l:
                continue
            row_arr = np.concatenate(rows_l)
            col_arr = np.concatenate(cols_l)
            sc_arr  = np.concatenate(sc_l).astype(np.float32)
            pairs = pd.DataFrame({
                "s1_id":       s1_ids[row_arr],
                "cand_id":     pool_ids[col_arr],
                "cand_source": np.int8(tgt_src),
                "bs_tfidf_full": sc_arr,
            })
            all_pairs.append(pairs)
            mean_c = len(pairs) / len(s1_cty)
            print(
                f"    {country:30s}  src->{tgt_src}  "
                f"s1={len(s1_cty):>7,}  pool={len(pool_cty):>8,}"
                f"{'(fallback)' if fallback else '':10s}  "
                f"pairs={len(pairs):>8,}  mean/s1={mean_c:.1f}"
            )

    elapsed = time.perf_counter() - t0
    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_tfidf_full"])
    )
    print(f"\n  [tfidf_full][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    result["blk_tfidf_full"] = True
    return result


# ---------------------------------------------------------------------------
# Pass: rare_token (SI-3)
# ---------------------------------------------------------------------------

def _run_rare_token(
    split: str,
    records: pd.DataFrame,
    s1_active: pd.DataFrame,
) -> pd.DataFrame:
    """Inverted-index blocking on rare name_core tokens.

    A token is "rare" if its document frequency in the whole split is <= RARE_DF_THRESH.
    Tokens that are pure digits or <= 2 characters are skipped.
    Score = max IDF of shared rare tokens.  Cap RARE_TOKEN_CAP per S1 per target source.
    Returns DataFrame[s1_id, cand_id, cand_source, bs_rare_token].
    """
    import math
    t0 = time.perf_counter()

    # --- Build per-split IDF (log((N+1)/(df+1)) + 1, sklearn convention) ---
    # Tokenise on whitespace; skip digits-only and short tokens
    def _tokenise(text: str) -> list[str]:
        return [
            tok for tok in str(text).split()
            if len(tok) > 2 and not tok.isdigit()
        ]

    N = len(records)  # total docs in this split
    # Build document-frequency map
    df_map: dict[str, int] = {}
    for text in records["name_core"]:
        for tok in set(_tokenise(text)):  # set: count each token once per doc
            df_map[tok] = df_map.get(tok, 0) + 1

    # IDF (sklearn smooth IDF)
    idf_map = {
        tok: math.log((N + 1) / (df + 1)) + 1
        for tok, df in df_map.items()
        if df <= RARE_DF_THRESH
    }
    rare_tokens = set(idf_map)
    print(
        f"  [rare_token] rare_tokens={len(rare_tokens):,}  active_s1={len(s1_active):,}"
    )

    # Pre-tokenise all records; keep only rare tokens per doc
    records = records.copy()
    records["_rtoks"] = records["name_core"].apply(
        lambda t: [tok for tok in _tokenise(t) if tok in rare_tokens]
    )

    countries = s1_active["country"].unique()
    all_pairs: list[pd.DataFrame] = []

    for country in countries:
        s1_cty = s1_active[s1_active["country"] == country]
        if len(s1_cty) == 0:
            continue
        # Merge rare tokens into s1_cty
        s1_cty = s1_cty.merge(
            records[["entity_id", "_rtoks"]],
            on="entity_id", how="left"
        )
        s1_cty["_rtoks"] = s1_cty["_rtoks"].apply(lambda x: x if isinstance(x, list) else [])

        for tgt_src in (2, 3):
            pool_cty = records[
                (records["source"] == tgt_src) & (records["country"] == country)
            ]
            fallback = False
            if len(pool_cty) == 0:
                pool_cty = records[records["source"] == tgt_src]
                fallback = True
            if len(pool_cty) == 0:
                continue

            # Build inverted index: token -> list of pool row indices
            inv_idx: dict[str, list[int]] = {}
            for idx, rtoks in enumerate(pool_cty["_rtoks"]):
                for tok in rtoks:
                    inv_idx.setdefault(tok, []).append(idx)
            pool_ids = pool_cty["entity_id"].values

            rows_out, cols_out, sc_out = [], [], []
            for s1_idx, row in enumerate(s1_cty.itertuples(index=False)):
                rtoks = row._rtoks  # pylint: disable=protected-access
                if not rtoks:
                    continue
                # Accumulate max-IDF per matched pool record
                hit_score: dict[int, float] = {}
                for tok in rtoks:
                    idf_val = idf_map.get(tok, 0.0)
                    for pool_idx in inv_idx.get(tok, []):
                        if hit_score.get(pool_idx, 0.0) < idf_val:
                            hit_score[pool_idx] = idf_val
                if not hit_score:
                    continue
                # Top-cap by score
                items = sorted(hit_score.items(), key=lambda kv: -kv[1])[:RARE_TOKEN_CAP]
                for pool_idx, sc in items:
                    rows_out.append(s1_idx)
                    cols_out.append(pool_idx)
                    sc_out.append(sc)

            if not rows_out:
                continue
            s1_id_arr = s1_cty["entity_id"].values
            pairs = pd.DataFrame({
                "s1_id":       s1_id_arr[rows_out],
                "cand_id":     pool_ids[cols_out],
                "cand_source": np.int8(tgt_src),
                "bs_rare_token": np.array(sc_out, dtype=np.float32),
            })
            all_pairs.append(pairs)
            mean_c = len(pairs) / len(s1_cty)
            print(
                f"    {country:30s}  src->{tgt_src}  "
                f"s1={len(s1_cty):>7,}  pool={len(pool_cty):>8,}"
                f"{'(fallback)' if fallback else '':10s}  "
                f"pairs={len(pairs):>8,}  mean/s1={mean_c:.1f}"
            )

    elapsed = time.perf_counter() - t0
    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_rare_token"])
    )
    print(f"\n  [rare_token][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    result["blk_rare_token"] = True
    return result


# ---------------------------------------------------------------------------
# Pass: addr_key (SI-3)
# ---------------------------------------------------------------------------

def _run_addr_key(
    split: str,
    records: pd.DataFrame,   # must include postcode, addr_numbers, city, addr_norm
    s1_active: pd.DataFrame,
) -> pd.DataFrame:
    """Exact-match blocking on two address keys.

    Key A: (postcode, first addr_numbers entry) when both are non-empty.
    Key B: (city, first addr_numbers entry, first addr_norm token) when all non-empty.
    Cap ADDR_KEY_CAP per S1 per target source.  Score = 1.0 (match is boolean).
    Returns DataFrame[s1_id, cand_id, cand_source, bs_addr_key].
    """
    t0 = time.perf_counter()

    def _first(lst: list) -> str:
        return lst[0] if lst else ""

    def _first_token(text: str) -> str:
        toks = str(text).split()
        return toks[0] if toks else ""

    # Pre-compute keys for all records
    records = records.copy()
    records["_ak_a"] = [
        (str(pc), str(fn))
        for pc, an in zip(records["postcode"], records["addr_numbers"])
        for fn in [_first(an) if isinstance(an, list) else ""]
    ]
    records["_ak_a"] = list(zip(
        records["postcode"].astype(str),
        records["addr_numbers"].apply(lambda x: _first(x) if isinstance(x, list) else ""),
    ))
    records["_ak_b"] = list(zip(
        records["city"].astype(str),
        records["addr_numbers"].apply(lambda x: _first(x) if isinstance(x, list) else ""),
        records["addr_norm"].apply(_first_token),
    ))

    countries = s1_active["country"].unique()
    all_pairs: list[pd.DataFrame] = []
    print(f"  [addr_key]  active_s1={len(s1_active):,}")

    for country in countries:
        s1_cty = s1_active[s1_active["country"] == country]
        if len(s1_cty) == 0:
            continue
        # Attach keys to s1
        s1_keys = s1_cty.merge(records[["entity_id", "_ak_a", "_ak_b"]], on="entity_id", how="left")

        for tgt_src in (2, 3):
            pool_cty = records[
                (records["source"] == tgt_src) & (records["country"] == country)
            ]
            fallback = False
            if len(pool_cty) == 0:
                pool_cty = records[records["source"] == tgt_src]
                fallback = True
            if len(pool_cty) == 0:
                continue

            # Build inverted index for each key type
            inv_a: dict[tuple, list[str]] = {}
            inv_b: dict[tuple, list[str]] = {}
            for row in pool_cty.itertuples(index=False):
                ka = row._ak_a
                kb = row._ak_b
                if ka[0] and ka[1]:      # postcode + first-number both non-empty
                    inv_a.setdefault(ka, []).append(row.entity_id)
                if kb[0] and kb[1] and kb[2]:  # city + number + street-token all non-empty
                    inv_b.setdefault(kb, []).append(row.entity_id)

            hit_pairs: dict[tuple[str, str], float] = {}
            for row in s1_keys.itertuples(index=False):
                s1_id = row.entity_id
                ka = row._ak_a
                kb = row._ak_b
                cands_hit: list[str] = []
                if ka[0] and ka[1]:
                    cands_hit.extend(inv_a.get(ka, []))
                if kb[0] and kb[1] and kb[2]:
                    cands_hit.extend(inv_b.get(kb, []))
                # Deduplicate and cap
                seen: set[str] = set()
                count = 0
                for cid in cands_hit:
                    if cid not in seen:
                        seen.add(cid)
                        hit_pairs[(s1_id, cid)] = 1.0
                        count += 1
                        if count >= ADDR_KEY_CAP:
                            break

            if not hit_pairs:
                continue
            s_ids, c_ids, sc_s = zip(*[(s, c, v) for (s, c), v in hit_pairs.items()])
            pairs = pd.DataFrame({
                "s1_id":       list(s_ids),
                "cand_id":     list(c_ids),
                "cand_source": np.int8(tgt_src),
                "bs_addr_key": np.array(sc_s, dtype=np.float32),
            })
            all_pairs.append(pairs)
            mean_c = len(pairs) / len(s1_cty)
            print(
                f"    {country:30s}  src->{tgt_src}  "
                f"s1={len(s1_cty):>7,}  pool={len(pool_cty):>8,}"
                f"{'(fallback)' if fallback else '':10s}  "
                f"pairs={len(pairs):>8,}  mean/s1={mean_c:.1f}"
            )

    elapsed = time.perf_counter() - t0
    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_addr_key"])
    )
    print(f"\n  [addr_key][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    result["blk_addr_key"] = True
    return result


# ---------------------------------------------------------------------------
# Pass: dense (SI-3)
# ---------------------------------------------------------------------------

def _run_dense(split: str) -> pd.DataFrame:
    """Load pre-computed dense neighbours from Adithya's embeddings pass.

    Reads cache/emb/dense_neighbors_{split}.parquet with columns [s1_id, cand_id, score].
    Returns DataFrame[s1_id, cand_id, cand_source, bs_dense] or empty if not found.
    """
    t0 = time.perf_counter()
    dense_path = config.cache_path(f"emb/dense_neighbors_{split}.parquet")
    if not dense_path.exists():
        print(
            f"  [dense] WARNING: {dense_path} not found – dense pass skipped."
        )
        return pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_dense"])

    dense = pd.read_parquet(dense_path, columns=["s1_id", "cand_id", "score"])
    dense = dense.drop_duplicates(["s1_id", "cand_id"]).reset_index(drop=True)
    # Infer cand_source from cand_id prefix
    dense["cand_source"] = dense["cand_id"].str[1].astype(np.int8)
    dense = dense.rename(columns={"score": "bs_dense"})
    dense["bs_dense"] = dense["bs_dense"].astype(np.float32)
    dense["blk_dense"] = True
    elapsed = time.perf_counter() - t0
    print(
        f"  [dense][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(dense):,}"
    )
    return dense[["s1_id", "cand_id", "cand_source", "bs_dense", "blk_dense"]]


# ---------------------------------------------------------------------------
# Pass: reverse (SI-3)
# ---------------------------------------------------------------------------

def _run_reverse(
    split: str,
    records: pd.DataFrame,
    s1_active: pd.DataFrame,
    vect_name: TfidfVectorizer,
) -> pd.DataFrame:
    """For every S2/S3 record, find its top REVERSE_K S1 by tfidf_name cosine.

    Uses the already-fitted tfidf_name vectoriser (same country grouping).
    Returns DataFrame[s1_id, cand_id, cand_source, bs_reverse].
    """
    t0 = time.perf_counter()
    countries = s1_active["country"].unique()
    all_pairs: list[pd.DataFrame] = []
    print(f"  [reverse]   active_s1={len(s1_active):,}")

    for country in countries:
        s1_cty = s1_active[s1_active["country"] == country]
        if len(s1_cty) == 0:
            continue
        s1_mat  = vect_name.transform(s1_cty["name_core"].values)
        s1_ids  = s1_cty["entity_id"].values

        for tgt_src in (2, 3):
            pool_cty = records[
                (records["source"] == tgt_src) & (records["country"] == country)
            ]
            if len(pool_cty) == 0:
                continue
            pool_mat = vect_name.transform(pool_cty["name_core"].values)
            pool_ids = pool_cty["entity_id"].values

            # Reverse: pool is the query, S1 is the pool
            # sim[i,j] = cosine(pool_i, s1_j)  -> top REVERSE_K s1 per pool record
            rows_l, cols_l, sc_l = [], [], []
            for ri, ci, sc in _top_k_per_row_chunk(pool_mat, s1_mat, REVERSE_K):
                rows_l.append(ri); cols_l.append(ci); sc_l.append(sc)
            if not rows_l:
                continue
            row_arr = np.concatenate(rows_l)   # pool indices
            col_arr = np.concatenate(cols_l)   # s1 indices
            sc_arr  = np.concatenate(sc_l).astype(np.float32)

            # Swap: s1_id is the S1 entry, cand_id is the pool (target) entry
            pairs = pd.DataFrame({
                "s1_id":       s1_ids[col_arr],
                "cand_id":     pool_ids[row_arr],
                "cand_source": np.int8(tgt_src),
                "bs_reverse":  sc_arr,
            })
            all_pairs.append(pairs)
            mean_c = len(pairs) / len(pool_cty)
            print(
                f"    {country:30s}  src->{tgt_src}  "
                f"s1={len(s1_cty):>7,}  pool={len(pool_cty):>8,}  "
                f"pairs={len(pairs):>8,}  mean/pool={mean_c:.1f}"
            )

    elapsed = time.perf_counter() - t0
    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_reverse"])
    )
    print(f"\n  [reverse][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    result["blk_reverse"] = True
    return result


# ---------------------------------------------------------------------------
# Union + pruning helpers (SI-3)
# ---------------------------------------------------------------------------

def _union_passes(pass_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge all per-pass DataFrames into one row per (s1_id, cand_id).

    blk_* columns are OR-ed; bs_* columns take the first non-NaN value found.
    """
    if not pass_dfs:
        # Return empty frame with all INTERFACES columns
        cols = (
            ["s1_id", "cand_id", "cand_source"]
            + [f"blk_{p}" for p in PASSES]
            + [f"bs_{p}" for p in PASSES]
        )
        return pd.DataFrame(columns=cols)

    # Ensure every df has a cand_source column
    for df in pass_dfs:
        if "cand_source" not in df.columns:
            df["cand_source"] = df["cand_id"].str[1].astype(np.int8)

    merged = pd.concat(pass_dfs, ignore_index=True)

    # Fill missing blk/bs columns with defaults before groupby
    for p in PASSES:
        if f"blk_{p}" not in merged.columns:
            merged[f"blk_{p}"] = False
        if f"bs_{p}" not in merged.columns:
            merged[f"bs_{p}"] = np.nan

    # Group by pair key: OR booleans, take first non-NaN for scores
    blk_cols = [f"blk_{p}" for p in PASSES]
    bs_cols  = [f"bs_{p}"  for p in PASSES]

    # cand_source: use first non-NaN (all rows for the same pair should agree)
    agg_dict: dict = {"cand_source": "first"}
    for col in blk_cols:
        agg_dict[col] = "any"
    for col in bs_cols:
        agg_dict[col] = "first"  # first non-NaN via sort later

    # Sort so that the row with the actual score is first in each group
    for col in bs_cols:
        merged[col] = merged[col].astype("float32")
    for col in blk_cols:
        merged[col] = merged[col].fillna(False).astype(bool)

    # For bs_* take max (best available score) rather than first
    # to ensure the strongest signal is kept when the same pair appears twice.
    agg_bs = {col: "max" for col in bs_cols}
    agg_blk = {col: "any" for col in blk_cols}
    agg_src = {"cand_source": "first"}

    result = merged.groupby(["s1_id", "cand_id"], as_index=False).agg(
        {**agg_src, **agg_blk, **agg_bs}
    )
    # Restore types
    for col in blk_cols:
        result[col] = result[col].astype(bool)
    for col in bs_cols:
        result[col] = result[col].astype(np.float32)
    result["cand_source"] = result["cand_source"].astype(np.int8)
    return result


def _prune(cands: pd.DataFrame, top_n: int, min_passes: int = 3) -> pd.DataFrame:
    """Prune to top_n per S1 using within-pass percentile-ranked prune_score.

    prune_score = max over active passes of (score's within-pass percentile rank,
    0-1).  Pairs found by >= min_passes passes are always kept.
    Sets prune_rank (0 = best within S1) and prune_score columns.
    """
    bs_cols  = [f"bs_{p}" for p in PASSES]
    blk_cols = [f"blk_{p}" for p in PASSES]

    # Compute within-pass percentile rank for each bs column (NaN -> 0)
    pct_cols: list[str] = []
    for p in PASSES:
        bs_col  = f"bs_{p}"
        pct_col = f"_pct_{p}"
        valid = cands[bs_col].notna()
        if valid.any():
            cands[pct_col] = np.nan
            cands.loc[valid, pct_col] = (
                cands.loc[valid, bs_col]
                .rank(pct=True, method="average")
                .values
            )
            cands[pct_col] = cands[pct_col].fillna(0.0)
        else:
            cands[pct_col] = 0.0
        pct_cols.append(pct_col)

    cands["prune_score"] = cands[pct_cols].max(axis=1).astype(np.float32)
    # Clean up temp columns
    cands = cands.drop(columns=pct_cols)

    # Number of passes that found each pair
    cands["_n_passes"] = cands[blk_cols].sum(axis=1)

    # Keep pairs always: rank within S1 by prune_score (0 = best)
    cands["prune_rank"] = (
        cands
        .groupby("s1_id")["prune_score"]
        .rank(method="first", ascending=False)
        .sub(1)
        .astype(np.int16)
    )

    # Retain top_n per S1 OR pairs found by >= min_passes passes
    cands = cands[
        (cands["prune_rank"] < top_n) | (cands["_n_passes"] >= min_passes)
    ].drop(columns=["_n_passes"]).reset_index(drop=True)

    return cands


# ---------------------------------------------------------------------------
# Main entry-point: generate_candidates (SI-3 version)
# ---------------------------------------------------------------------------

def generate_candidates(split: str, top_n: int = 50) -> pd.DataFrame:
    """Union all blocking passes, prune to top_n per S1, save candidates_{split}.parquet.

    Implements all six passes (SI-1 + SI-3):
      tfidf_name, tfidf_full, rare_token, addr_key, dense, reverse
    prune_score is the max within-pass percentile rank across passes.
    Pairs found by >= 3 passes are always kept regardless of top_n.
    """
    t_total = time.perf_counter()
    print(f"\n=== generate_candidates(split={split!r}, top_n={top_n}) ===")

    records = _load_records_full(split)
    src_counts = records["source"].value_counts().sort_index().to_dict()
    print(f"  Loaded {len(records):,} records  {src_counts}")

    # Active S1 (dev-sample or all)
    s1_all = records[records["source"] == 1].copy()
    if config.SAMPLE_FRAC < 1.0:
        s1_active = s1_all[s1_all["entity_id"].map(io_utils.in_dev_sample)].copy()
    else:
        s1_active = s1_all.copy()
    print(f"  Active S1: {len(s1_active):,}  (SAMPLE_FRAC={config.SAMPLE_FRAC})")

    # ---- Fit shared name vectoriser (used by tfidf_name AND reverse) ----
    vect_name = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
    )
    vect_name.fit(records["name_core"].values)
    print(f"  Shared name TF-IDF vocab: {len(vect_name.vocabulary_):,}")

    # ---- Run all passes ----
    pass_dfs: list[pd.DataFrame] = []

    # 1. tfidf_name (uses shared vectoriser)
    print("\n-- Pass: tfidf_name --")
    df_name = _run_tfidf_name_with_vect(split, records, s1_active, vect_name)
    pass_dfs.append(df_name)

    # 2. tfidf_full
    print("\n-- Pass: tfidf_full --")
    df_full = _run_tfidf_full(split, records, s1_active, vect_name)
    pass_dfs.append(df_full)

    # 3. rare_token
    print("\n-- Pass: rare_token --")
    df_rare = _run_rare_token(split, records, s1_active)
    pass_dfs.append(df_rare)

    # 4. addr_key
    print("\n-- Pass: addr_key --")
    df_addr = _run_addr_key(split, records, s1_active)
    pass_dfs.append(df_addr)

    # 5. dense
    print("\n-- Pass: dense --")
    df_dense = _run_dense(split)
    pass_dfs.append(df_dense)

    # 6. reverse
    print("\n-- Pass: reverse --")
    df_rev = _run_reverse(split, records, s1_active, vect_name)
    pass_dfs.append(df_rev)

    # ---- Union ----
    print("\n-- Union passes --")
    cands = _union_passes(pass_dfs)
    print(f"  Union: {len(cands):,} unique pairs before pruning")

    # ---- Prune ----
    cands = _prune(cands, top_n=top_n)
    print(f"  After pruning (top_n={top_n}): {len(cands):,} pairs")

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
        f"\n  Saved {len(cands):,} candidate pairs -> {out_path}"
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


def report(split: str = "train", top_n: int | None = None) -> dict:
    """Load cache/candidates_{split}.parquet, compute and print blocking diagnostics.

    Only meaningful for split='train' (ground truth available).
    top_n: if provided, simulate pruning at this N before computing metrics
           (uses the prune_rank column already in the parquet).
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

    cands_full = pd.read_parquet(cands_path)
    print(f"\n=== blocking report (split={split!r}) ===")
    print(f"  Loaded {len(cands_full):,} candidate pairs from {cands_path}")

    # Optionally restrict to top_n per S1 (using existing prune_rank)
    if top_n is not None and "prune_rank" in cands_full.columns:
        cands = cands_full[cands_full["prune_rank"] < top_n].copy()
        print(f"  Restricted to prune_rank < {top_n}: {len(cands):,} pairs")
    else:
        cands = cands_full

    # Load ground truth
    gt_raw = io_utils.read_ground_truth()
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
    print(f"  OVERALL  (top_n={top_n if top_n is not None else 'all'})")
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
        "date_ist", "split", "top_n", "passes", "tfidf_k", "chunk_size",
        "total_pairs", "pair_recall", "entity_ceiling", "oracle_f05",
        "mean_cands", "p50_cands", "p95_cands",
    ]
    row = {
        "date_ist":       datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "split":          split,
        "top_n":          top_n if top_n is not None else "all",
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


def sweep_top_n(
    split: str = "train",
    ns: tuple[int, ...] = (20, 30, 50, 80),
) -> None:
    """Run report() at each value of N in *ns* using the already-saved candidates.

    Prints a compact summary table of recall vs mean-candidates so the team
    can pick the best N for the final pruning step.
    Uses prune_rank from the parquet, so no re-generation is needed.
    Each run also appends a row to docs/blocking_log.csv.
    """
    print(f"\n{'='*72}")
    print(f"  N-SWEEP  (split={split!r},  N in {list(ns)})")
    print(f"{'='*72}")
    print(f"  {'N':>5}  {'pairs':>10}  {'recall':>8}  {'entity_ceil':>12}  "
          f"{'oracle_f05':>11}  {'mean':>7}  {'p95':>6}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*12}  {'-'*11}  {'-'*7}  {'-'*6}")

    for n in ns:
        try:
            s = report(split=split, top_n=n)
        except FileNotFoundError as exc:
            print(f"  {n:>5}  ERROR: {exc}")
            continue
        print(
            f"  {n:>5}  {s['total_pairs']:>10,}  "
            f"{s['pair_recall']:>8.4f}  "
            f"{s['entity_ceiling']:>12.4f}  "
            f"{s['oracle_f05']:>11.4f}  "
            f"{s['mean_cands']:>7.1f}  "
            f"{s['p95_cands']:>6.0f}"
        )

    print(f"{'='*72}")
    print(f"  Rows appended to docs/blocking_log.csv")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate blocking candidates for one split and optionally report."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--top-n", type=int, default=50, dest="top_n",
        help="Prune to this many candidates per S1 (default: 50).",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="After generating, run the blocking report (train only).",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run report at N in [20,30,50,80] to compare recall vs candidates.",
    )
    parser.add_argument(
        "--report-only", action="store_true", dest="report_only",
        help="Skip generation; only run report/sweep on existing candidates file.",
    )
    args = parser.parse_args()

    if not args.report_only:
        generate_candidates(args.split, top_n=args.top_n)

    if args.sweep:
        sweep_top_n(args.split)
    elif args.report:
        report(args.split, top_n=args.top_n)

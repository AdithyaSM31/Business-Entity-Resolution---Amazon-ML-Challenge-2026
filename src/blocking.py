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

PASSES = ("dense", "exact_name", "rare_token", "addr_key", "tfidf_name", "tfidf_full", "reverse")
ALL_PASSES = PASSES
DEFAULT_PASSES = ("dense", "exact_name", "rare_token", "addr_key")

# Default caps per S1 per target source
EXACT_NAME_CAP = 30
ADDR_KEY_CAP  = 30
RARE_TOKEN_CAP = 30

# Rare-token inverted-index threshold
RARE_DF_THRESH = 20

# Optional TF-IDF passes constants (opt-in via --passes)
TFIDF_K = 20
CHUNK_SIZE = 2_000
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
# Pass: exact_name (SI-3 / v1)
# ---------------------------------------------------------------------------

def _run_exact_name(
    split: str,
    records: pd.DataFrame,
    s1_active: pd.DataFrame,
    cap: int = EXACT_NAME_CAP,
) -> pd.DataFrame:
    """Exact-match blocking on normalized business name (name_core).

    Matches S1 against target pools (sources 2 and 3) on exact name_core,
    partitioned by country. Fully vectorized with pandas merges.
    Capped at `cap` candidates per S1 per target source. Score = 1.0.
    Returns DataFrame[s1_id, cand_id, cand_source, bs_exact_name, blk_exact_name].
    """
    t0 = time.perf_counter()
    s1_valid = s1_active[s1_active["name_core"].str.strip() != ""][
        ["entity_id", "country", "name_core"]
    ].rename(columns={"entity_id": "s1_id"})

    all_pairs: list[pd.DataFrame] = []
    for tgt_src in (2, 3):
        pool = records[
            (records["source"] == tgt_src) & (records["name_core"].str.strip() != "")
        ][["entity_id", "country", "name_core"]].rename(columns={"entity_id": "cand_id"})

        if len(s1_valid) == 0 or len(pool) == 0:
            continue

        merged = s1_valid.merge(pool, on=["country", "name_core"], how="inner")
        if len(merged) == 0:
            continue

        if cap is not None:
            merged["_rank"] = merged.groupby("s1_id").cumcount()
            merged = merged[merged["_rank"] < cap].drop(columns=["_rank"])

        merged["cand_source"] = np.int8(tgt_src)
        merged["bs_exact_name"] = np.float32(1.0)
        merged["blk_exact_name"] = True
        all_pairs.append(
            merged[["s1_id", "cand_id", "cand_source", "bs_exact_name", "blk_exact_name"]]
        )

    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_exact_name", "blk_exact_name"])
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  [exact_name][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    return result


# ---------------------------------------------------------------------------
# Pass: rare_token (SI-3 / vectorized)
# ---------------------------------------------------------------------------

def _run_rare_token(
    split: str,
    records: pd.DataFrame,
    s1_active: pd.DataFrame,
    cap: int = RARE_TOKEN_CAP,
) -> pd.DataFrame:
    """Inverted-index blocking on rare name_core tokens (vectorized).

    A token is "rare" if its document frequency in the whole split is <= RARE_DF_THRESH.
    Tokens that are pure digits or <= 2 characters are skipped.
    Score = max IDF of shared rare tokens. Capped at `cap` per S1 per target source.
    Vectorized: explodes token IDs and performs an inner merge on (country, token_id).
    Returns DataFrame[s1_id, cand_id, cand_source, bs_rare_token, blk_rare_token].
    """
    import math
    t0 = time.perf_counter()

    def _tokenise(text: str) -> list[str]:
        return [tok for tok in str(text).split() if len(tok) > 2 and not tok.isdigit()]

    N = len(records)
    rec_tokens = records["name_core"].apply(_tokenise)
    token_counts = pd.Series([tok for toks in rec_tokens for tok in set(toks)]).value_counts()
    rare_tokens = set(token_counts[token_counts <= RARE_DF_THRESH].index)

    idf_map = {
        tok: math.log((N + 1) / (token_counts[tok] + 1)) + 1.0
        for tok in rare_tokens
    }
    print(f"  [rare_token] rare_tokens={len(rare_tokens):,}  active_s1={len(s1_active):,}")
    if not rare_tokens:
        return pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_rare_token", "blk_rare_token"])

    tok_to_id = {tok: idx for idx, tok in enumerate(rare_tokens)}
    tok_id_to_idf = np.array([idf_map[tok] for tok in rare_tokens], dtype=np.float32)

    # Build exploded table for active S1
    s1_rec = s1_active[["entity_id", "country", "name_core"]].copy()
    s1_rec["toks"] = s1_rec["name_core"].apply(
        lambda t: [tok_to_id[x] for x in _tokenise(t) if x in tok_to_id]
    )
    s1_exploded = s1_rec.explode("toks").dropna(subset=["toks"])
    if len(s1_exploded) == 0:
        return pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_rare_token", "blk_rare_token"])

    s1_exploded["toks"] = s1_exploded["toks"].astype(np.int32)
    s1_exploded = (
        s1_exploded.rename(columns={"entity_id": "s1_id", "toks": "token_id"})[
            ["s1_id", "country", "token_id"]
        ].drop_duplicates()
    )

    all_pairs: list[pd.DataFrame] = []
    for tgt_src in (2, 3):
        pool = records[records["source"] == tgt_src][["entity_id", "country", "name_core"]].copy()
        pool["toks"] = pool["name_core"].apply(
            lambda t: [tok_to_id[x] for x in _tokenise(t) if x in tok_to_id]
        )
        pool_exploded = pool.explode("toks").dropna(subset=["toks"])
        if len(pool_exploded) == 0:
            continue
        pool_exploded["toks"] = pool_exploded["toks"].astype(np.int32)
        pool_exploded = (
            pool_exploded.rename(columns={"entity_id": "cand_id", "toks": "token_id"})[
                ["cand_id", "country", "token_id"]
            ].drop_duplicates()
        )

        merged = s1_exploded.merge(pool_exploded, on=["country", "token_id"], how="inner")
        if len(merged) == 0:
            continue

        merged["score"] = tok_id_to_idf[merged["token_id"].values]
        pair_scores = merged.groupby(["s1_id", "cand_id"], as_index=False)["score"].max()

        pair_scores = pair_scores.sort_values(["s1_id", "score"], ascending=[True, False])
        pair_scores["_rank"] = pair_scores.groupby("s1_id").cumcount()
        pair_scores = pair_scores[pair_scores["_rank"] < cap].drop(columns=["_rank"])

        pair_scores["cand_source"] = np.int8(tgt_src)
        pair_scores["bs_rare_token"] = pair_scores["score"].astype(np.float32)
        pair_scores["blk_rare_token"] = True
        all_pairs.append(
            pair_scores[["s1_id", "cand_id", "cand_source", "bs_rare_token", "blk_rare_token"]]
        )

    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_rare_token", "blk_rare_token"])
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  [rare_token][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
    return result


# ---------------------------------------------------------------------------
# Pass: addr_key (SI-3 / vectorized)
# ---------------------------------------------------------------------------

def _run_addr_key(
    split: str,
    records: pd.DataFrame,
    s1_active: pd.DataFrame,
    cap: int = ADDR_KEY_CAP,
) -> pd.DataFrame:
    """Exact-match blocking on two address keys (vectorized).

    Key A: (postcode, first addr_numbers entry) when both are non-empty. Score = 1.0.
    Key B: (city, first addr_numbers entry, first addr_norm token) when all non-empty. Score = 0.8.
    Capped at `cap` per S1 per target source.
    Returns DataFrame[s1_id, cand_id, cand_source, bs_addr_key, blk_addr_key].
    """
    t0 = time.perf_counter()

    def _first(lst: list) -> str:
        return str(lst[0]).strip() if (isinstance(lst, list) and lst) else ""

    def _first_token(text: str) -> str:
        toks = str(text).strip().split()
        return toks[0] if toks else ""

    # Ensure necessary columns exist
    for col in ["postcode", "city", "addr_norm"]:
        if col not in records.columns:
            records[col] = ""
    if "addr_numbers" not in records.columns:
        records["addr_numbers"] = [[] for _ in range(len(records))]

    pc = records["postcode"].astype(str).str.strip()
    fn = records["addr_numbers"].apply(_first)
    city = records["city"].astype(str).str.strip().str.lower()
    stok = records["addr_norm"].apply(_first_token).str.lower()

    has_a = (pc != "") & (fn != "")
    has_b = (city != "") & (fn != "") & (stok != "")

    key_a = (pc + "_" + fn).where(has_a, "")
    key_b = (city + "_" + fn + "_" + stok).where(has_b, "")

    records_keys = pd.DataFrame({
        "entity_id": records["entity_id"],
        "source": records["source"],
        "country": records["country"],
        "key_a": key_a,
        "key_b": key_b,
    })

    s1_keys = s1_active[["entity_id", "country"]].merge(
        records_keys[["entity_id", "key_a", "key_b"]], on="entity_id", how="left"
    ).rename(columns={"entity_id": "s1_id"})

    all_pairs: list[pd.DataFrame] = []
    for tgt_src in (2, 3):
        pool_src = records_keys[records_keys["source"] == tgt_src].rename(
            columns={"entity_id": "cand_id"}
        )

        # Match Key A
        s1_a = s1_keys[s1_keys["key_a"] != ""][["s1_id", "country", "key_a"]]
        pool_a = pool_src[pool_src["key_a"] != ""][["cand_id", "country", "key_a"]]
        merged_a = s1_a.merge(pool_a, on=["country", "key_a"], how="inner")[["s1_id", "cand_id"]]
        merged_a["score"] = np.float32(1.0)

        # Match Key B
        s1_b = s1_keys[s1_keys["key_b"] != ""][["s1_id", "country", "key_b"]]
        pool_b = pool_src[pool_src["key_b"] != ""][["cand_id", "country", "key_b"]]
        merged_b = s1_b.merge(pool_b, on=["country", "key_b"], how="inner")[["s1_id", "cand_id"]]
        merged_b["score"] = np.float32(0.8)

        combined = pd.concat([merged_a, merged_b], ignore_index=True)
        if len(combined) == 0:
            continue

        combined = combined.sort_values(["s1_id", "score"], ascending=[True, False]).drop_duplicates(
            ["s1_id", "cand_id"]
        )
        combined["_rank"] = combined.groupby("s1_id").cumcount()
        combined = combined[combined["_rank"] < cap].drop(columns=["_rank"])

        combined["cand_source"] = np.int8(tgt_src)
        combined["bs_addr_key"] = combined["score"].astype(np.float32)
        combined["blk_addr_key"] = True
        all_pairs.append(
            combined[["s1_id", "cand_id", "cand_source", "bs_addr_key", "blk_addr_key"]]
        )

    result = (
        pd.concat(all_pairs, ignore_index=True).drop_duplicates(["s1_id", "cand_id"])
        if all_pairs
        else pd.DataFrame(columns=["s1_id", "cand_id", "cand_source", "bs_addr_key", "blk_addr_key"])
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  [addr_key][{split}]  elapsed={elapsed:.1f}s  total_pairs={len(result):,}")
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
# SI-6 (1): Learned pruning
# ---------------------------------------------------------------------------

def learned_prune(
    split: str = "train",
    top_n: int = 50,
    n_splits: int = 5,
    min_passes: int = 3,
) -> pd.DataFrame:
    """Fit a LogisticRegression on blocking features to replace the heuristic prune_score.

    Uses GroupKFold(n_splits) grouped by s1_id so no S1's candidates appear in
    both train and validation folds.  The final prune_score written to the
    parquet is the OOF probability on train; on test a single model is fit on
    all train data and applied.

    Features: bs_* filled with -1 when NaN, blk_* as 0/1, n_passes.
    Target:   ground-truth label (1 = true match).

    After fitting, re-prunes at the requested top_n with the new scores and
    calls sweep_top_n() to compare recall.  Returns the updated candidates DF.

    Parameters
    ----------
    split     : 'train' or 'test'
    top_n     : number of candidates to keep per S1 after re-pruning
    n_splits  : GroupKFold splits (train only)
    min_passes: pairs found by >= min_passes are always kept (same as _prune)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    print(f"\n=== learned_prune(split={split!r}, top_n={top_n}) ===")

    cands_path = config.cache_path(f"candidates_{split}.parquet")
    if not cands_path.exists():
        raise FileNotFoundError(
            f"candidates_{split}.parquet not found – run generate_candidates first."
        )
    cands = pd.read_parquet(cands_path)
    print(f"  Loaded {len(cands):,} candidate pairs")

    bs_cols  = [f"bs_{p}"  for p in PASSES]
    blk_cols = [f"blk_{p}" for p in PASSES]

    # Build feature matrix
    feat_parts = []
    for col in bs_cols:
        if col in cands.columns:
            feat_parts.append(cands[col].fillna(-1.0).astype(np.float32).values.reshape(-1, 1))
        else:
            feat_parts.append(np.full((len(cands), 1), -1.0, dtype=np.float32))
    for col in blk_cols:
        if col in cands.columns:
            feat_parts.append(cands[col].fillna(False).astype(np.float32).values.reshape(-1, 1))
        else:
            feat_parts.append(np.zeros((len(cands), 1), dtype=np.float32))
    n_passes_vec = np.zeros((len(cands), 1), dtype=np.float32)
    for col in blk_cols:
        if col in cands.columns:
            n_passes_vec[:, 0] += cands[col].fillna(False).astype(np.float32).values
    feat_parts.append(n_passes_vec)

    X = np.hstack(feat_parts)
    feat_names = bs_cols + blk_cols + ["n_passes"]
    print(f"  Feature matrix: {X.shape}")

    if split == "train":
        # Need ground truth labels
        gt_raw = io_utils.read_ground_truth()
        if config.SAMPLE_FRAC < 1.0:
            gt_raw = gt_raw[gt_raw["s1_id"].map(io_utils.in_dev_sample)].copy()
        gt_set: set[tuple[str, str]] = set()
        for row in gt_raw.itertuples(index=False):
            for m in row.matches:
                gt_set.add((row.s1_id, m))
        y = np.array(
            [1 if (s, c) in gt_set else 0
             for s, c in zip(cands["s1_id"], cands["cand_id"])],
            dtype=np.int8,
        )
        print(f"  Labels: {y.sum():,} positives / {len(y):,} total  "
              f"({y.mean()*100:.2f}% positive rate)")

        groups = cands["s1_id"].values
        oof_prob = np.zeros(len(cands), dtype=np.float32)

        gkf = GroupKFold(n_splits=n_splits)
        for fold_i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_va = scaler.transform(X[va_idx])
            clf = LogisticRegression(
                max_iter=500, C=1.0, class_weight="balanced", solver="lbfgs"
            )
            clf.fit(X_tr, y[tr_idx])
            oof_prob[va_idx] = clf.predict_proba(X_va)[:, 1]
            va_auc = _roc_auc_simple(y[va_idx], oof_prob[va_idx])
            print(f"  fold {fold_i+1}/{n_splits}  val_auc={va_auc:.4f}")

        cands["prune_score"] = oof_prob.astype(np.float32)

        # Also fit a final model on all data and save coefficients for test
        scaler_all = StandardScaler()
        X_all = scaler_all.fit_transform(X)
        clf_all = LogisticRegression(
            max_iter=500, C=1.0, class_weight="balanced", solver="lbfgs"
        )
        clf_all.fit(X_all, y)

        # Persist model parameters for test-time use
        import json
        model_dir = config.cache_path("learned_prune")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_params = {
            "coef": clf_all.coef_.tolist(),
            "intercept": clf_all.intercept_.tolist(),
            "scaler_mean": scaler_all.mean_.tolist(),
            "scaler_scale": scaler_all.scale_.tolist(),
            "feature_names": feat_names,
        }
        params_path = model_dir / "lr_params.json"
        params_path.write_text(json.dumps(model_params, indent=2))
        print(f"  Saved LR params -> {params_path}")

        # Feature importances (absolute |coef|)
        coef = np.abs(clf_all.coef_[0])
        top_feat = sorted(zip(feat_names, coef), key=lambda x: -x[1])[:10]
        print("\n  Top-10 features by |coef|:")
        for fname, fcoef in top_feat:
            print(f"    {fname:<20s}  {fcoef:.4f}")

    else:  # split == 'test'
        # Load model params fitted on train
        import json
        params_path = config.cache_path("learned_prune") / "lr_params.json"
        if not params_path.exists():
            raise FileNotFoundError(
                f"{params_path} not found – run learned_prune('train') first."
            )
        mp = json.loads(params_path.read_text())
        mean  = np.array(mp["scaler_mean"],  dtype=np.float32)
        scale = np.array(mp["scaler_scale"], dtype=np.float32)
        coef  = np.array(mp["coef"][0],      dtype=np.float32)
        intercept = float(mp["intercept"][0])
        X_scaled  = (X - mean) / scale
        logit     = X_scaled @ coef + intercept
        prob      = 1.0 / (1.0 + np.exp(-logit))
        cands["prune_score"] = prob.astype(np.float32)

    # Re-rank and re-prune
    blk_cols_present = [c for c in blk_cols if c in cands.columns]
    cands["_n_passes"] = cands[blk_cols_present].fillna(False).sum(axis=1)
    cands["prune_rank"] = (
        cands.groupby("s1_id")["prune_score"]
        .rank(method="first", ascending=False)
        .sub(1)
        .astype(np.int16)
    )
    cands = cands[
        (cands["prune_rank"] < top_n) | (cands["_n_passes"] >= min_passes)
    ].drop(columns=["_n_passes"]).reset_index(drop=True)
    print(f"\n  After learned prune (top_n={top_n}): {len(cands):,} pairs")

    # Save updated candidates
    cands.to_parquet(cands_path, index=False)
    print(f"  Saved updated candidates -> {cands_path}")

    # Re-run sweep if on train
    if split == "train":
        print("\n  Re-running sweep_top_n to compare recall after learned pruning:")
        sweep_top_n(split)

    return cands


def _roc_auc_simple(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC without scikit-learn dependency at call time."""
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Main entry-point: generate_candidates (SI-3 version)
# ---------------------------------------------------------------------------

def generate_candidates(
    split: str,
    top_n: int = 30,
    passes: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """Generate and prune candidates for *split*.

    Default passes: dense, exact_name, rare_token, addr_key.
    Optional passes (via passes arg): tfidf_name, tfidf_full, reverse.
    """
    t_total = time.perf_counter()
    if passes is None:
        active_passes = list(DEFAULT_PASSES)
    else:
        active_passes = [p for p in passes if p in ALL_PASSES]

    print(f"\n=== generate_candidates(split={split!r}, top_n={top_n}, passes={active_passes}) ===")

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

    # Only fit TF-IDF vectoriser if any TF-IDF pass was explicitly requested
    need_tfidf_vect = any(p in active_passes for p in ("tfidf_name", "tfidf_full", "reverse"))
    vect_name = None
    if need_tfidf_vect:
        vect_name = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True
        )
        vect_name.fit(records["name_core"].values)
        print(f"  Shared name TF-IDF vocab: {len(vect_name.vocabulary_):,}")

    pass_dfs: list[pd.DataFrame] = []

    # 1. dense
    if "dense" in active_passes:
        print("\n-- Pass: dense --")
        df_dense = _run_dense(split)
        pass_dfs.append(df_dense)

    # 2. exact_name
    if "exact_name" in active_passes:
        print("\n-- Pass: exact_name --")
        df_exact = _run_exact_name(split, records, s1_active)
        pass_dfs.append(df_exact)

    # 3. rare_token
    if "rare_token" in active_passes:
        print("\n-- Pass: rare_token --")
        df_rare = _run_rare_token(split, records, s1_active)
        pass_dfs.append(df_rare)

    # 4. addr_key
    if "addr_key" in active_passes:
        print("\n-- Pass: addr_key --")
        df_addr = _run_addr_key(split, records, s1_active)
        pass_dfs.append(df_addr)

    # 5. tfidf_name (optional)
    if "tfidf_name" in active_passes and vect_name is not None:
        print("\n-- Pass: tfidf_name --")
        df_name = _run_tfidf_name_with_vect(split, records, s1_active, vect_name)
        pass_dfs.append(df_name)

    # 6. tfidf_full (optional)
    if "tfidf_full" in active_passes and vect_name is not None:
        print("\n-- Pass: tfidf_full --")
        df_full = _run_tfidf_full(split, records, s1_active, vect_name)
        pass_dfs.append(df_full)

    # 7. reverse (optional)
    if "reverse" in active_passes and vect_name is not None:
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
    if config.SAMPLE_FRAC < 1.0:
        gt_raw = gt_raw[gt_raw["s1_id"].map(io_utils.in_dev_sample)].copy()
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
# SI-6 (2): Missed pairs analysis
# ---------------------------------------------------------------------------

def misses_report(split: str = "train", examples: int = 10) -> pd.DataFrame:
    """Analyse ground-truth pairs that NO blocking pass retrieved.

    For each missed pair, loads the raw name_core, addr_norm, postcode, city
    for both S1 and the target record, then groups misses into five categories:

      1. name_very_different  – normalised edit-distance(name_core_s1, name_core_cand) > 0.6
      2. address_only_match   – name very different but share a non-empty address key
      3. empty_fields         – either side has empty name_core or empty addr_norm
      4. cross_country        – s1 and cand have different country strings
      5. other                – everything else

    Prints counts and *examples* random examples per group.
    Returns a DataFrame of all missed pairs with their group label.
    """
    if split != "train":
        print("[misses_report] Ground truth only available for split='train'; skipping.")
        return pd.DataFrame()

    print(f"\n=== misses_report(split={split!r}) ===")

    # Load candidates
    cands_path = config.cache_path(f"candidates_{split}.parquet")
    if not cands_path.exists():
        raise FileNotFoundError(f"{cands_path} not found – run generate_candidates first.")
    cand_pairs: set[tuple[str, str]] = set(
        zip(*pd.read_parquet(cands_path, columns=["s1_id", "cand_id"]).values.T)
    )
    print(f"  Candidate pairs: {len(cand_pairs):,}")

    # Load ground truth
    gt_raw = io_utils.read_ground_truth()
    if config.SAMPLE_FRAC < 1.0:
        gt_raw = gt_raw[gt_raw["s1_id"].map(io_utils.in_dev_sample)].copy()
    gt_pairs: list[tuple[str, str]] = []
    for row in gt_raw.itertuples(index=False):
        for m in row.matches:
            gt_pairs.append((row.s1_id, m))
    print(f"  GT pairs: {len(gt_pairs):,}")

    missed = [(s, c) for s, c in gt_pairs if (s, c) not in cand_pairs]
    print(f"  Missed:   {len(missed):,}  ({len(missed)/len(gt_pairs)*100:.2f}% of GT)")
    if not missed:
        print("  No misses – blocking is perfect!")
        return pd.DataFrame()

    # Load records for context
    recs_path = config.cache_path("records.parquet")
    want = ["split", "source", "entity_id", "country",
            "name_core", "addr_norm", "postcode", "city", "addr_numbers"]
    if recs_path.exists():
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(recs_path).names)
        cols = [c for c in want if c in avail]
        recs = pd.read_parquet(recs_path, columns=cols)
        recs = recs[recs["split"] == split] if "split" in recs.columns else recs
    else:
        frames = []
        for src in config.SOURCES:
            raw = io_utils.read_source(split, src)
            raw["name_core"] = raw["business_name"].str.lower().fillna("")
            raw["source"] = src
            frames.append(raw)
        recs = pd.concat(frames, ignore_index=True)

    for col in ["addr_norm", "postcode", "city", "country", "name_core"]:
        if col not in recs.columns:
            recs[col] = ""
        else:
            recs[col] = recs[col].fillna("").astype(str)

    rec_idx = recs.set_index("entity_id")

    # Build missed DataFrame
    s1_ids  = [s for s, _ in missed]
    cnd_ids = [c for _, c in missed]

    def _get(col: str, ids: list[str]) -> list[str]:
        return rec_idx[col].reindex(ids).fillna("").tolist()

    df = pd.DataFrame({
        "s1_id":         s1_ids,
        "cand_id":       cnd_ids,
        "country_s1":    _get("country",   s1_ids),
        "country_cand":  _get("country",   cnd_ids),
        "name_s1":       _get("name_core", s1_ids),
        "name_cand":     _get("name_core", cnd_ids),
        "addr_s1":       _get("addr_norm", s1_ids),
        "addr_cand":     _get("addr_norm", cnd_ids),
        "postcode_s1":   _get("postcode",  s1_ids),
        "postcode_cand": _get("postcode",  cnd_ids),
        "city_s1":       _get("city",      s1_ids),
        "city_cand":     _get("city",      cnd_ids),
    })

    # Normalised edit distance (Levenshtein / max-len)
    try:
        from rapidfuzz.distance import Levenshtein
        def _ned(a: str, b: str) -> float:
            if not a and not b:
                return 0.0
            mx = max(len(a), len(b))
            return Levenshtein.distance(a, b) / mx if mx else 0.0
    except ImportError:
        def _ned(a: str, b: str) -> float:  # type: ignore[misc]
            if a == b:
                return 0.0
            if not a or not b:
                return 1.0
            # Cheap approximation: Jaccard on chars
            sa, sb = set(a), set(b)
            return 1.0 - len(sa & sb) / len(sa | sb)

    df["ned_name"] = [
        _ned(n1, n2) for n1, n2 in zip(df["name_s1"], df["name_cand"])
    ]

    def _shares_addr(row: pd.Series) -> bool:
        """True if postcode matches and both are non-empty."""
        return bool(row["postcode_s1"] and row["postcode_s1"] == row["postcode_cand"])

    df["_shares_addr"] = df.apply(_shares_addr, axis=1)

    NAME_DIFF_THRESH = 0.6

    def _group(row: pd.Series) -> str:
        if row["country_s1"] != row["country_cand"]:
            return "cross_country"
        if not row["name_s1"] or not row["name_cand"]:
            return "empty_fields"
        if row["ned_name"] > NAME_DIFF_THRESH:
            if row["_shares_addr"]:
                return "address_only_match"
            return "name_very_different"
        return "other"

    df["group"] = df.apply(_group, axis=1)

    # --- Print counts ---
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  MISSED PAIR GROUPS  (total={len(df):,})")
    print(sep)
    counts = df["group"].value_counts()
    for grp, cnt in counts.items():
        print(f"  {grp:<25s}  {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")
    print(sep)

    # --- Print examples ---
    display_cols = ["s1_id", "cand_id", "country_s1", "country_cand",
                    "name_s1", "name_cand", "addr_s1", "addr_cand", "ned_name"]
    for grp in counts.index:
        grp_df = df[df["group"] == grp]
        sample = grp_df.sample(min(examples, len(grp_df)), random_state=42)
        print(f"\n--- {grp}  (n={len(grp_df):,}, showing {len(sample)}) ---")
        with pd.option_context("display.max_colwidth", 60, "display.width", 200):
            print(sample[display_cols].to_string(index=False))

    # Save to CSV for later inspection
    out_path = config.ROOT / "docs" / "misses_report.csv"
    df.drop(columns=["_shares_addr"]).to_csv(out_path, index=False)
    print(f"\n  Full miss list -> {out_path}")
    return df


# ---------------------------------------------------------------------------
# SI-6 (3): France / distribution shift report
# ---------------------------------------------------------------------------

def france_report(
    split: str = "test",
    countries: tuple[str, ...] = ("US", "India", "France"),
) -> pd.DataFrame:
    """Compare blocking quality across countries for a given split.

    Prints, per country:
      - Number of active S1s
      - Mean / median / p95 candidates per S1
      - Distribution of bs_tfidf_name for the BEST candidate per S1
        (mean, p25, p50, p75, p95, fraction with score < 0.3)

    Flags countries where the best-candidate score distribution looks shifted
    (mean < 0.5 or > 20% with score < 0.3).

    Parameters
    ----------
    split     : which split to analyse (default 'test')
    countries : which country strings to highlight (others shown as 'other')
    """
    print(f"\n=== france_report(split={split!r}) ===")

    cands_path = config.cache_path(f"candidates_{split}.parquet")
    if not cands_path.exists():
        raise FileNotFoundError(f"{cands_path} not found – run generate_candidates first.")

    want_cols = ["s1_id", "cand_id"]
    if "bs_tfidf_name" not in pd.read_parquet(cands_path, columns=["s1_id"]).columns:
        # peek at actual columns
        sample_df = pd.read_parquet(cands_path)
        actual_cols = list(sample_df.columns)
        del sample_df
    else:
        actual_cols = None

    cands = pd.read_parquet(cands_path)
    bs_col = "bs_tfidf_name" if "bs_tfidf_name" in cands.columns else None
    print(f"  Loaded {len(cands):,} candidate pairs")

    # Attach country from records
    recs_path = config.cache_path("records.parquet")
    if recs_path.exists():
        import pyarrow.parquet as pq
        avail = set(pq.read_schema(recs_path).names)
        cols = [c for c in ["split", "source", "entity_id", "country"] if c in avail]
        recs = pd.read_parquet(recs_path, columns=cols)
        if "split" in recs.columns:
            recs = recs[recs["split"] == split]
        s1_country = (
            recs[recs["source"] == 1][["entity_id", "country"]]
            .rename(columns={"entity_id": "s1_id", "country": "_country"})
        )
    else:
        raw_s1 = io_utils.read_source(split, 1)
        s1_country = raw_s1[["entity_id", "country"]].rename(
            columns={"entity_id": "s1_id", "country": "_country"}
        )
    cands = cands.merge(s1_country, on="s1_id", how="left")
    cands["_country"] = cands["_country"].fillna("unknown")

    # Best bs_tfidf_name per S1
    if bs_col:
        best_bs = (
            cands.groupby("s1_id")[["_country", bs_col]]
            .apply(lambda g: pd.Series({
                "country": g["_country"].iloc[0],
                "best_bs":  g[bs_col].max(),
            }))
            .reset_index()
        )
    else:
        best_bs = (
            cands.groupby("s1_id")["_country"]
            .first()
            .reset_index()
            .rename(columns={"_country": "country"})
        )
        best_bs["best_bs"] = np.nan

    n_cands_per_s1 = cands.groupby("s1_id")["cand_id"].count().rename("n_cands")
    best_bs = best_bs.join(n_cands_per_s1, on="s1_id")

    # Country bucketing
    cty_set = set(countries)
    best_bs["_cty_label"] = best_bs["country"].apply(
        lambda c: c if c in cty_set else "other"
    )

    sep  = "-" * 80
    sep2 = "=" * 80
    print(f"\n{sep2}")
    print(f"  DISTRIBUTION SHIFT REPORT  (split={split!r})")
    print(f"{sep2}")
    hdr = (f"  {'Country':<20} {'S1s':>8} {'mean_cands':>11} "
           f"{'p50_cands':>10} {'p95_cands':>10} "
           f"{'mean_bs':>8} {'p25_bs':>7} {'p50_bs':>7} "
           f"{'p75_bs':>7} {'p95_bs':>7} {'<0.3':>6} {'FLAG':>5}")
    print(hdr)
    print(sep)

    rows = []
    # Print the requested countries first, then 'other'
    labels_order = list(countries) + ["other"]
    seen = set()
    for label in labels_order:
        if label in seen:
            continue
        seen.add(label)
        grp = best_bs[best_bs["_cty_label"] == label]
        if len(grp) == 0:
            continue
        n_s1       = len(grp)
        mean_cands = grp["n_cands"].mean()
        p50_cands  = grp["n_cands"].quantile(0.50)
        p95_cands  = grp["n_cands"].quantile(0.95)
        if bs_col and grp["best_bs"].notna().any():
            mean_bs = grp["best_bs"].mean()
            p25_bs  = grp["best_bs"].quantile(0.25)
            p50_bs  = grp["best_bs"].quantile(0.50)
            p75_bs  = grp["best_bs"].quantile(0.75)
            p95_bs  = grp["best_bs"].quantile(0.95)
            frac_low = (grp["best_bs"] < 0.3).mean()
        else:
            mean_bs = p25_bs = p50_bs = p75_bs = p95_bs = float("nan")
            frac_low = float("nan")
        flag = "(!)" if (mean_bs < 0.5 or frac_low > 0.20) else ""
        print(
            f"  {label:<20} {n_s1:>8,} {mean_cands:>11.1f} "
            f"{p50_cands:>10.0f} {p95_cands:>10.0f} "
            f"{mean_bs:>8.3f} {p25_bs:>7.3f} {p50_bs:>7.3f} "
            f"{p75_bs:>7.3f} {p95_bs:>7.3f} "
            f"{frac_low:>6.2f} {flag:>5}"
        )
        rows.append({
            "country": label, "n_s1": n_s1,
            "mean_cands": mean_cands, "p50_cands": p50_cands, "p95_cands": p95_cands,
            "mean_bs": mean_bs, "p25_bs": p25_bs, "p50_bs": p50_bs,
            "p75_bs": p75_bs, "p95_bs": p95_bs, "frac_low_bs": frac_low, "flag": flag,
        })
    print(sep2)
    print("  NOTE: '(!)' = mean_bs < 0.5 or >20% of S1s have best-cand bs_tfidf_name < 0.3")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate blocking candidates for one split and optionally report."
    )
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument(
        "--top-n", type=int, default=30, dest="top_n",
        help="Prune to this many candidates per S1 (default: 30).",
    )
    parser.add_argument(
        "--passes", nargs="+", default=None,
        help="Which passes to run (e.g. dense exact_name rare_token addr_key). Default: dense exact_name rare_token addr_key.",
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
    # SI-6 flags
    parser.add_argument(
        "--learned-prune", action="store_true", dest="learned_prune",
        help="(SI-6) Fit LR pruner on blocking features and re-prune (train only).",
    )
    parser.add_argument(
        "--misses", action="store_true",
        help="(SI-6) Print analysis of GT pairs not found by any pass (train only).",
    )
    parser.add_argument(
        "--france", action="store_true",
        help="(SI-6) Print per-country distribution-shift report.",
    )
    parser.add_argument(
        "--france-split", default=None, dest="france_split",
        help="Split to use for --france (default: same as --split).",
    )
    args = parser.parse_args()

    if not args.report_only:
        generate_candidates(args.split, top_n=args.top_n, passes=args.passes)

    if args.learned_prune:
        learned_prune(args.split, top_n=args.top_n)

    if args.sweep:
        sweep_top_n(args.split)
    elif args.report:
        report(args.split, top_n=args.top_n)

    if args.misses:
        misses_report(args.split)

    if args.france:
        france_split = args.france_split or args.split
        france_report(france_split)

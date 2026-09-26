"""TF-IDF baseline submission for AR-2.

For each Source 1 entity, retrieve the best S2 and best S3 record in the same
country using char_wb TF-IDF cosine similarity on lowercased ``name address``.
The train threshold is tuned on the deterministic development sample.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from src import config, io_utils, metrics

NGRAM_RANGE = (3, 5)
CHUNK_SIZE = 2_000
THRESHOLDS = np.round(np.arange(0.30, 0.951, 0.01), 2)


def _text(df: pd.DataFrame) -> pd.Series:
    return (df["business_name"].astype(str) + " " + df["business_address"].astype(str)).str.lower()


def _active_s1(df: pd.DataFrame, split: str, s1_limit: int | None = None) -> pd.DataFrame:
    if split == "train" and config.SAMPLE_FRAC < 1.0:
        df = df[df["entity_id"].map(io_utils.in_dev_sample)].copy()
    else:
        df = df.copy()
    if s1_limit is not None:
        df = df.head(s1_limit).copy()
    return df

def _top_matches(s1: pd.DataFrame, target: pd.DataFrame, k: int) -> pd.DataFrame:
    if s1.empty or target.empty:
        return pd.DataFrame(columns=["s1_id", "cand_id", "score"])

    # Fit on this same-country corpus so comparisons stay within the country and
    # the vectorizer never has to materialise a cross-country candidate matrix.
    corpus = pd.concat([_text(s1), _text(target)], ignore_index=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=NGRAM_RANGE,
                                 min_df=1, sublinear_tf=True)
    vectorizer.fit(corpus)
    q = vectorizer.transform(_text(s1))
    x = vectorizer.transform(_text(target))

    n_neighbors = min(k, len(target))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute",
                          n_jobs=config.N_JOBS)
    nn.fit(x)

    rows = []
    for start in range(0, len(s1), CHUNK_SIZE):
        q_chunk = q[start:start + CHUNK_SIZE]
        distances, indices = nn.kneighbors(q_chunk, return_distance=True)
        scores = 1.0 - distances
        for i in range(q_chunk.shape[0]):
            for j in range(n_neighbors):
                rows.append((s1.iloc[start + i]["entity_id"],
                             target.iloc[indices[i, j]]["entity_id"],
                             float(scores[i, j])))
    return pd.DataFrame(rows, columns=["s1_id", "cand_id", "score"])


def retrieve(split: str, k: int, s1_limit: int | None = None, target_limit: int | None = None) -> pd.DataFrame:
    """Retrieve top-k per target source for every active S1, grouped by country."""
    s1 = _active_s1(io_utils.read_source(split, 1), split, s1_limit)
    s2 = io_utils.read_source(split, 2)
    s3 = io_utils.read_source(split, 3)
    frames = []

    countries = s1["country"].astype(str).unique()
    for country in countries:
        q = s1[s1["country"].astype(str) == country]
        for source, target in ((2, s2), (3, s3)):
            pool = target[target["country"].astype(str) == country]
            if target_limit is not None:
                pool = pool.head(target_limit)
            if pool.empty or q.empty:
                continue
            result = _top_matches(q, pool, k)
            if not result.empty:
                result["cand_source"] = np.int8(source)
                frames.append(result)
        print(f"{split}: country={country!r}, S1={len(q):,}")

    if not frames:
        return pd.DataFrame(columns=["s1_id", "cand_id", "score", "cand_source"])
    return pd.concat(frames, ignore_index=True)


def _truth_dict() -> dict[str, list[str]]:
    gt = io_utils.read_ground_truth()
    return dict(zip(gt["s1_id"], gt["matches"]))


def tune_threshold(best_train: pd.DataFrame, active_s1_ids: list[str]) -> tuple[float, float]:
    truth_all = _truth_dict()
    truth = {s1_id: truth_all.get(s1_id, []) for s1_id in active_s1_ids}
    score_by_s1 = {s1_id: {} for s1_id in active_s1_ids}
    for s1_id, group in best_train.groupby("s1_id", sort=False):
        score_by_s1[s1_id] = group.set_index("cand_id")["score"].to_dict()

    best_t, best_score = float(THRESHOLDS[0]), -1.0
    print("threshold\tmacro_f05")
    for threshold in THRESHOLDS:
        pred = {}
        for s1_id, scores in score_by_s1.items():
            pred[s1_id] = [cand_id for cand_id, score in scores.items() if score >= threshold]
        score = float(metrics.macro_f05(pred, truth))
        print(f"{threshold:.2f}\t{score:.6f}")
        # Prefer the higher threshold on ties because F0.5 penalizes false matches.
        if score > best_score or (score == best_score and threshold > best_t):
            best_t, best_score = float(threshold), score
    print(f"best_threshold={best_t:.2f}, train_macro_f05={best_score:.6f}")
    return best_t, best_score


def _lists_from_candidates(candidates: pd.DataFrame, s1_ids: list[str], threshold: float,
                           top_k: int = 10) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    matched: dict[str, list[str]] = {}
    candidate_ids: dict[str, list[str]] = {}
    for s1_id, group in candidates.groupby("s1_id", sort=False):
        ordered = group.sort_values(["cand_source", "score", "cand_id"], ascending=[True, False, True])
        candidate_ids[s1_id] = []
        matched[s1_id] = []
        for source in (2, 3):
            source_rows = ordered[ordered["cand_source"] == source]
            candidate_ids[s1_id].extend(source_rows["cand_id"].head(top_k).tolist())
            if not source_rows.empty and float(source_rows.iloc[0]["score"]) >= threshold:
                matched[s1_id].append(source_rows.iloc[0]["cand_id"])
    return matched, candidate_ids


def run() -> None:
    t0 = time.perf_counter()
    train_s1_ids = _active_s1(io_utils.read_source("train", 1), "train")["entity_id"].tolist()
    train_best = retrieve("train", k=1)
    threshold, _ = tune_threshold(train_best, train_s1_ids)

    test_top10 = retrieve("test", k=10)
    s1_ids = io_utils.read_source("test", 1)["entity_id"].tolist()
    matched, candidate_ids = _lists_from_candidates(test_top10, s1_ids, threshold, top_k=10)

    # Ensure every test S1 is represented, including countries with no same-country target.
    for s1_id in s1_ids:
        matched.setdefault(s1_id, [])
        candidate_ids.setdefault(s1_id, [])

    io_utils.write_id_lists(s1_ids, matched, "matched_entity_ids", config.OUTPUT_DIR / "matching_results.tsv")
    io_utils.write_id_lists(s1_ids, candidate_ids, "candidate_entity_ids", config.OUTPUT_DIR / "candidate_pairs.tsv")
    print(f"wrote outputs under {config.OUTPUT_DIR}")

    validator = Path(__file__).resolve().parent / "validate.py"
    result = subprocess.run([
        sys.executable, str(validator),
        "--matching", str(config.OUTPUT_DIR / "matching_results.tsv"),
        "--candidate", str(config.OUTPUT_DIR / "candidate_pairs.tsv"),
        "--test-dir", str(config.DATA_DIR / "test"),
    ])
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"elapsed={time.perf_counter() - t0:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AR-2 TF-IDF baseline.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Run both for the normal baseline; train/test are useful for debugging.")
    parser.add_argument("--s1-limit", type=int, default=None,
                    help="Testing only: limit the number of Source 1 records.")
    parser.add_argument("--target-limit", type=int, default=None,
                    help="Testing only: limit Source 2/3 candidates per country.")
    args = parser.parse_args()
    if args.split == "train":
        train_s1 = _active_s1(io_utils.read_source("train", 1), "train", args.s1_limit)
        train_ids = train_s1["entity_id"].tolist()
        train_best = retrieve("train", k=1, s1_limit=args.s1_limit,
                            target_limit=args.target_limit)
        tune_threshold(train_best, train_ids)
        return

    if args.split == "test":
        retrieve("test", k=10, s1_limit=args.s1_limit,
                target_limit=args.target_limit)
        return
    run()


if __name__ == "__main__":
    main()

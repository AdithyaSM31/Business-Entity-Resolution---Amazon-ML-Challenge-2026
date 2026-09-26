"""Quick stage-1 model on the dense candidates (AD-3a, v2). Owner: Adithya.

A self-contained first submission that needs nothing from the other stages:
candidates = the dense neighbours (train: dense_neighbors_train_all.parquet filtered to the 20% sample,
so every S1 competes against the full field of S1s exactly as on test), features = dense score, rank
and competition features plus rapidfuzz name/address scores on basic-cleaned text, model = LightGBM
trained 5-fold on cache/folds.parquet (out-of-fold predictions for every sampled S1, saved in the
format src/evaluate.py reads), selection = exclusivity + a threshold tuned on those predictions.

It is a stop-gap: the proper pipeline (Siva's candidates, Bhanu's features, Arushi's folds, metric,
writers and selection) replaces each piece as it lands.

Usage:
  python -m src.quick_model text                      # cache basic-cleaned name/address (both splits)
  python -m src.quick_model pairs --split train       # candidates + competition features; then --split test
  python -m src.quick_model features --split train    # then --split test
  python -m src.quick_model train --run-id <id>
  python -m src.quick_model submit --run-id <id>      # predict test, write output/*.tsv, validate
"""
import argparse
import json
import re
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import config, io_utils
from .embeddings import _basic_clean

POSTCODE_RE = r"(?<!\d)(\d{5,6})(?!\d)"
FIRST_NUMBER_RE = r"(\d+)"
PAIR_CHUNK = 2_000_000

NAME_SCORERS = {
    "ratio": fuzz.ratio,
    "token_set": fuzz.token_set_ratio,
    "token_sort": fuzz.token_sort_ratio,
    "partial": fuzz.partial_ratio,
    "jw": JaroWinkler.normalized_similarity,
}
ADDR_SCORERS = {"ratio": fuzz.ratio, "token_set": fuzz.token_set_ratio}


def _log(msg: str) -> None:
    print(f"[quick_model {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------ text cache

def build_text(split: str) -> None:
    """cache/quick/text_{split}.parquet: entity_id, source, name, addr (basic-cleaned), postcode, num."""
    parts = []
    for source in config.SOURCES:
        df = io_utils.read_source(split, source)
        name = df["business_name"].map(_basic_clean)
        addr = df["business_address"].str.replace(r"<\s*null\s*>|\bnull\b|\bn/a\b", " ", case=False, regex=True)
        parts.append(pd.DataFrame({
            "entity_id": df["entity_id"], "source": np.int8(source), "name": name,
            "addr": addr.map(_basic_clean),
            "postcode": addr.str.findall(POSTCODE_RE).str[-1].fillna(""),
            "num": addr.str.extract(FIRST_NUMBER_RE, expand=False).fillna(""),
            "non_latin": df["business_name"].map(lambda s: not s.isascii()).astype(np.int8),
        }))
        _log(f"text {split} S{source}: {len(df):,} records")
    pd.concat(parts, ignore_index=True).to_parquet(config.cache_path(f"quick/text_{split}.parquet"), index=False)


# ------------------------------------------------------------------------------------------ features

def _cpdist(a: np.ndarray, b: np.ndarray, scorer) -> np.ndarray:
    out = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    return out if scorer is JaroWinkler.normalized_similarity else out / 100.0


def _group_rank(key: np.ndarray, score: np.ndarray):
    """Per row: rank of its score within its key group (0 = best), the group's best score, and the
    best score among the OTHER rows of the group (NaN when it is alone)."""
    order = np.lexsort((-score, key))
    k_sorted, s_sorted = key[order], score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(k_sorted)) + 1]
    sizes = np.diff(np.r_[starts, len(order)])
    start_of = np.repeat(starts, sizes)
    rank, best, other = (np.empty(len(key), np.float32) for _ in range(3))
    rank[order] = np.arange(len(order)) - start_of
    best[order] = s_sorted[start_of]
    second = np.where(np.repeat(sizes, sizes) > 1, s_sorted[np.minimum(start_of + 1, len(order) - 1)], np.nan)
    other[order] = np.where(rank[order] == 0, second, s_sorted[start_of])
    return rank, best, other


def build_pairs(split: str) -> None:
    """cache/quick/pairs_{split}.parquet: candidate pairs, grouped by S1, with competition features.

    Competition features (how many S1s list this record, where this S1 ranks among them, the best rival
    score) are computed on the full field of S1s: for train that is dense_neighbors_train_all, which is
    then filtered to the dev sample. Computing them on the sample alone makes every record look ~2.7x
    less contested on train than on test.
    """
    t0 = time.time()
    name = "emb/dense_neighbors_train_all.parquet" if split == "train" else f"emb/dense_neighbors_{split}.parquet"
    t = pq.read_table(config.cache_path(name), columns=["s1_id", "cand_id", "score"])
    s1 = pc.dictionary_encode(t["s1_id"]).combine_chunks()
    cand = pc.dictionary_encode(t["cand_id"]).combine_chunks()
    s_code = s1.indices.to_numpy().astype(np.int64)
    c_code = cand.indices.to_numpy().astype(np.int64)
    score = t["score"].to_numpy().astype(np.float32)
    _log(f"{split}: {len(score):,} pairs over {len(s1.dictionary):,} S1 loaded ({time.time() - t0:.0f}s)")

    rev_rank, cand_best, rival = _group_rank(c_code, score)
    s1_rank, _, _ = _group_rank(s_code, score)
    rev_n = np.bincount(c_code)[c_code].astype(np.float32)
    close = (score >= cand_best - 0.02).astype(np.float64)
    comp = {
        "q_rev_n": rev_n,
        "q_rev_rank": rev_rank,
        "q_rev_n_close": np.bincount(c_code, weights=close)[c_code].astype(np.float32),
        "q_cand_best": cand_best,
        "q_gap_cand_best": (cand_best - score).astype(np.float32),
        "q_rival_margin": (score - rival).astype(np.float32),  # >0 when this S1 beats every rival
        "q_mutual_best": ((s1_rank == 0) & (rev_rank == 0)).astype(np.float32),
    }
    keep = np.arange(len(score))
    if split == "train" and config.SAMPLE_FRAC < 1:
        in_sample = np.fromiter((io_utils.in_dev_sample(v) for v in s1.dictionary.to_pylist()),
                                bool, len(s1.dictionary))
        keep = np.flatnonzero(in_sample[s_code])
    out = pa.table({"s1_id": t["s1_id"].take(keep), "cand_id": t["cand_id"].take(keep),
                    "score": pa.array(score[keep]), **{k: pa.array(v[keep]) for k, v in comp.items()}})
    pq.write_table(out, config.cache_path(f"quick/pairs_{split}.parquet"))
    _log(f"{split}: saved {out.num_rows:,} pairs for {pc.count_distinct(out['s1_id']).as_py():,} S1 "
         f"({time.time() - t0:.0f}s)")


def _s1_chunks(path, batch_rows: int = PAIR_CHUNK):
    """Yield DataFrames of whole S1 groups (the dense file is grouped by S1), about batch_rows each."""
    carry = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_rows):
        df = batch.to_pandas()
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last = df["s1_id"].iloc[-1]
        cut = int(np.searchsorted((df["s1_id"] == last).to_numpy(), True))  # first row of the last S1
        if cut == 0:
            carry = df
            continue
        carry = df.iloc[cut:].reset_index(drop=True)
        yield df.iloc[:cut].reset_index(drop=True)
    if carry is not None and len(carry):
        yield carry


def build_features(split: str) -> None:
    """One row per dense-neighbour pair, built in whole-S1 chunks and streamed to parquet."""
    t0 = time.time()
    text = pd.read_parquet(config.cache_path(f"quick/text_{split}.parquet"))
    index = pd.Index(text["entity_id"])
    arrays = {c: text[c].to_numpy() for c in ("name", "addr", "postcode", "num", "non_latin")}
    del text
    _log(f"{split}: text cache loaded ({time.time() - t0:.0f}s)")
    path = config.cache_path(f"quick/pairs_{split}.parquet")
    writer, n_done = None, 0
    for pairs in _s1_chunks(path):
        feats = _chunk_features(pairs, index, arrays)
        table = pa.Table.from_pandas(feats, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(config.cache_path(f"quick/features_{split}.parquet"), table.schema)
        writer.write_table(table)
        n_done += len(feats)
        _log(f"{split}: {n_done:,} pairs done ({time.time() - t0:.0f}s)")
    writer.close()
    _log(f"{split}: saved {n_done:,} pairs in {time.time() - t0:.0f}s")


def _chunk_features(pairs: pd.DataFrame, index: pd.Index, arrays: dict) -> pd.DataFrame:
    """Dense score/rank features and string similarities for a chunk of whole S1 groups."""
    pairs["cand_source"] = pairs["cand_id"].str[1].astype(np.int8)
    g = pairs.groupby("s1_id")["score"]
    gs = pairs.groupby(["s1_id", "cand_source"])["score"]
    feats = pd.DataFrame({
        "s1_id": pairs["s1_id"], "cand_id": pairs["cand_id"],
        "q_dense": pairs["score"].astype(np.float32),
        "q_cand_source": pairs["cand_source"].astype(np.float32),
        "q_rank": g.rank(method="first", ascending=False).astype(np.float32),
        "q_rank_src": gs.rank(method="first", ascending=False).astype(np.float32),
        "q_gap_best": (g.transform("max") - pairs["score"]).astype(np.float32),
        "q_gap_best_src": (gs.transform("max") - pairs["score"]).astype(np.float32),
        "q_n_cands": g.transform("size").astype(np.float32),
        "q_n_close": pairs.assign(c=(pairs["score"] >= g.transform("max") - 0.02))
                          .groupby("s1_id")["c"].transform("sum").astype(np.float32),
        "q_mean_score": g.transform("mean").astype(np.float32),
    })
    del g, gs
    for col in pairs.columns:  # competition features from build_pairs
        if col.startswith("q_"):
            feats[col] = pairs[col].to_numpy()

    a_pos = index.get_indexer(feats["s1_id"])
    b_pos = index.get_indexer(feats["cand_id"])
    if (a_pos < 0).any() or (b_pos < 0).any():
        raise KeyError("some pair ids are missing from the text cache")
    name, addr = arrays["name"], arrays["addr"]
    post, num, non_latin = arrays["postcode"], arrays["num"], arrays["non_latin"]
    for k, scorer in NAME_SCORERS.items():
        feats[f"q_name_{k}"] = _cpdist(name[a_pos], name[b_pos], scorer)
    for k, scorer in ADDR_SCORERS.items():
        feats[f"q_addr_{k}"] = _cpdist(addr[a_pos], addr[b_pos], scorer)

    pa_, pb_ = post[a_pos], post[b_pos]
    na_, nb_ = num[a_pos], num[b_pos]
    both_pc = (pa_ != "") & (pb_ != "")
    both_num = (na_ != "") & (nb_ != "")
    feats["q_postcode"] = np.where(both_pc, (pa_ == pb_).astype(np.float32), np.nan).astype(np.float32)
    feats["q_number"] = np.where(both_num, (na_ == nb_).astype(np.float32), np.nan).astype(np.float32)
    feats["q_cand_addr_empty"] = (addr[b_pos] == "").astype(np.float32)
    feats["q_cand_non_latin"] = non_latin[b_pos].astype(np.float32)
    feats["q_name_len_ratio"] = (pd.Series(name[b_pos]).str.len().to_numpy()
                                 / np.maximum(pd.Series(name[a_pos]).str.len().to_numpy(), 1)).astype(np.float32)
    # best name score of this S1 per source, and this pair's gap to it (duplicates cluster together)
    best = feats.groupby(["s1_id", "q_cand_source"])["q_name_token_set"].transform("max")
    feats["q_name_gap_src"] = (best - feats["q_name_token_set"]).astype(np.float32)
    return feats


# ------------------------------------------------------------------------------------------ metric

def macro_f05(pred: pd.DataFrame, n_true: pd.Series) -> float:
    """pred: selected pairs with column `label`; n_true: true-match count for every scored S1 (index s1_id)."""
    per = pred.groupby("s1_id")["label"].agg(["sum", "size"])
    tp = per["sum"].reindex(n_true.index, fill_value=0).to_numpy()
    n_pred = per["size"].reindex(n_true.index, fill_value=0).to_numpy()
    nt = n_true.to_numpy()
    f = np.where((n_pred == 0) & (nt == 0), 1.0,
                 np.where((n_pred == 0) | (nt == 0), 0.0, 1.25 * tp / (0.25 * nt + np.maximum(n_pred, 1))))
    return float(f.mean())


def select(df: pd.DataFrame, threshold: float, exclusive: bool) -> pd.DataFrame:
    """Pairs with prob >= threshold; with exclusivity each cand_id keeps only its best S1."""
    sel = df[df["prob"] >= threshold]
    if exclusive:
        sel = sel.sort_values("prob", ascending=False).drop_duplicates("cand_id")
    return sel


# ------------------------------------------------------------------------------------------ train

PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_child_samples=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              num_threads=config.N_JOBS, verbose=-1, seed=config.SEED)


def train(run_id: str, max_rounds: int = 3000) -> None:
    """5-fold LightGBM on cache/folds.parquet, out-of-fold predictions saved for src/evaluate.py,
    a selection rule tuned on them, then one model on the whole sample for test."""
    import lightgbm as lgb

    t0 = time.time()
    df = pd.read_parquet(config.cache_path("quick/features_train.parquet"))
    features = [c for c in df.columns if c.startswith("q_")]
    labels = pd.read_parquet(config.cache_path("labels.parquet"))
    folds = pd.read_parquet(config.cache_path("folds.parquet"), columns=["s1_id", "n_true", "fold"])
    df = df.merge(labels, on=["s1_id", "cand_id"], how="left").merge(folds[["s1_id", "fold"]], on="s1_id", how="left")
    if df["fold"].isna().any():
        raise ValueError("some candidate S1 are missing from cache/folds.parquet; rerun python -m src.folds")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    n_true = folds.set_index("s1_id")["n_true"]
    found = df["label"].sum() / n_true.sum()
    _log(f"{len(df):,} pairs, {len(features)} features, positives {df['label'].mean():.3f}, "
         f"candidate recall {found:.4f} over {len(n_true):,} S1")

    oof = np.zeros(len(df), dtype=np.float32)
    best_iters = []
    for k in sorted(folds["fold"].unique()):
        va = (df["fold"] == k).to_numpy()
        dtr = lgb.Dataset(df.loc[~va, features], df.loc[~va, "label"])
        dva = lgb.Dataset(df.loc[va, features], df.loc[va, "label"], reference=dtr)
        model = lgb.train(PARAMS, dtr, max_rounds, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(500)])
        oof[va] = model.predict(df.loc[va, features], num_iteration=model.best_iteration)
        best_iters.append(model.best_iteration)
        _log(f"fold {k}: {model.best_iteration} rounds, logloss {model.best_score['valid_0']['binary_logloss']:.4f} "
             f"({time.time() - t0:.0f}s)")
    df["prob"] = oof
    df[["s1_id", "cand_id", "prob", "fold"]].to_parquet(config.cache_path(f"preds/{run_id}_train.parquet"), index=False)

    # exclusivity does not depend on the threshold, so apply it once
    view = df[["s1_id", "cand_id", "label", "prob", "q_dense"]]
    excl = view.sort_values("prob", ascending=False).drop_duplicates("cand_id")
    results = []
    for is_excl, base in ((False, view), (True, excl)):
        for t in np.arange(0.30, 0.86, 0.02):
            results.append((is_excl, round(float(t), 2), macro_f05(base[base["prob"] >= t], n_true)))
    best_excl, best_t, best_f = max(results, key=lambda r: r[2])
    dense_excl = view.sort_values("q_dense", ascending=False).drop_duplicates("cand_id")
    dense_ref = max(((round(float(t), 3), macro_f05(dense_excl[dense_excl["q_dense"] >= t], n_true))
                     for t in np.arange(0.88, 0.98, 0.005)), key=lambda r: r[1])
    per_fold = {int(k): macro_f05(
        (excl if best_excl else view).pipe(lambda d: d[(d["prob"] >= best_t) & d["s1_id"].isin(set(g))]),
        n_true.loc[g]) for k, g in folds.groupby("fold")["s1_id"]}
    _log(f"out-of-fold macro F0.5 {best_f:.4f} (threshold {best_t}, exclusive {best_excl}) over {len(n_true):,} S1; "
         f"per fold {', '.join(f'{v:.4f}' for v in per_fold.values())}; dense score only {dense_ref[1]:.4f}")

    rounds = int(np.mean(best_iters) * 1.1)
    final = lgb.train(PARAMS, lgb.Dataset(df[features], df["label"]), rounds)
    out = config.MODEL_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out / "model.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=features).sort_values(ascending=False)
    imp.to_csv(out / "importance.csv")
    meta = {"features": features, "threshold": best_t, "exclusive": best_excl, "oof_f05": best_f,
            "oof_f05_per_fold": per_fold, "dense_only_f05": dense_ref[1], "candidate_recall": float(found),
            "fold_rounds": best_iters, "rounds": rounds, "params": PARAMS, "grid": results}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "postprocess.json").write_text(json.dumps({"method": "threshold", "threshold": best_t,
                                                      "exclusive": best_excl, "per_source": False}, indent=1))
    _log(f"saved {out}; top features: {', '.join(imp.index[:10])}; total {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------ submit

def _write_lists(s1_order: np.ndarray, pairs: pd.DataFrame, list_col: str, path) -> None:
    lists = pairs.groupby("s1_id")["cand_id"].agg(",".join)
    col = pd.Series(s1_order).map(lists).fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{list_col}\n")
        for s1, ids in zip(s1_order, col):
            f.write(f"{s1}\t{ids}\n")


def submit(run_id: str) -> None:
    import lightgbm as lgb

    t0 = time.time()
    out = config.MODEL_DIR / run_id
    meta = json.loads((out / "meta.json").read_text())
    model = lgb.Booster(model_file=str(out / "model.txt"))
    pf = pq.ParquetFile(config.cache_path("quick/features_test.parquet"))
    probs = []
    for batch in pf.iter_batches(batch_size=PAIR_CHUNK, columns=["s1_id", "cand_id"] + meta["features"]):
        b = batch.to_pandas()
        probs.append(pd.DataFrame({"s1_id": b["s1_id"], "cand_id": b["cand_id"],
                                   "prob": model.predict(b[meta["features"]], num_threads=config.N_JOBS)
                                   .astype(np.float32)}))
    preds = pd.concat(probs, ignore_index=True)
    preds.to_parquet(config.cache_path(f"preds/{run_id}_test.parquet"), index=False)
    sel = select(preds, meta["threshold"], meta["exclusive"])
    s1_order = io_utils.read_source("test", 1)["entity_id"].to_numpy()
    _write_lists(s1_order, sel, "matched_entity_ids", config.OUTPUT_DIR / "matching_results.tsv")
    _write_lists(s1_order, preds, "candidate_entity_ids", config.OUTPUT_DIR / "candidate_pairs.tsv")
    n_s1 = sel["s1_id"].nunique()
    _log(f"test: {len(preds):,} candidate pairs, {len(sel):,} matches, {n_s1:,} of {len(s1_order):,} S1 "
         f"non-empty ({time.time() - t0:.0f}s)")
    validator = config.ROOT / "utils" / "validate_submission.py"
    if not validator.exists():
        validator = config.ROOT.parent / "student_resource" / "utils" / "validate_submission.py"
    subprocess.run([sys.executable, str(validator), "--matching", str(config.OUTPUT_DIR / "matching_results.tsv"),
                    "--candidate", str(config.OUTPUT_DIR / "candidate_pairs.tsv"),
                    "--test-dir", str(config.DATA_DIR / "test")], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["text", "pairs", "features", "train", "submit"])
    parser.add_argument("--split", choices=["train", "test"])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.step == "text":
        for split in ([args.split] if args.split else config.SPLITS):
            build_text(split)
    elif args.step == "pairs":
        build_pairs(args.split)
    elif args.step == "features":
        build_features(args.split)
    elif args.step == "train":
        train(args.run_id)
    else:
        submit(args.run_id)

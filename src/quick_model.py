"""Quick stage-1 model on the dense candidates (AD-3a). Owner: Adithya.

A self-contained first submission that needs nothing from the other stages:
candidates = cache/emb/dense_neighbors_{split}.parquet, features = dense score and rank features plus
rapidfuzz name/address scores on basic-cleaned text, model = LightGBM, selection = exclusivity + a
threshold tuned on a held-out 20% of the train sample for macro F0.5.

It is a stop-gap: the proper pipeline (Siva's candidates, Bhanu's features, Arushi's folds, metric,
writers and selection) replaces each piece as it lands.

Usage:
  python -m src.quick_model text                      # cache basic-cleaned name/address (both splits)
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
import zlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from . import config, io_utils
from .embeddings import _basic_clean

POSTCODE_RE = r"(?<!\d)(\d{5,6})(?!\d)"
FIRST_NUMBER_RE = r"(\d+)"
PAIR_CHUNK = 2_000_000
HOLDOUT_BUCKETS = 2  # hash(s1_id) % 10 < 2 -> held out (20% of the sample)

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
    path = config.cache_path(f"emb/dense_neighbors_{split}.parquet")
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

def _holdout(s1_ids: pd.Series) -> np.ndarray:
    return np.fromiter((zlib.crc32(s.encode()) % 10 < HOLDOUT_BUCKETS for s in s1_ids), bool, len(s1_ids))


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

FEATURES = None  # resolved from the parquet at train time


def train(run_id: str, rounds: int = 2000) -> None:
    import lightgbm as lgb

    t0 = time.time()
    df = pd.read_parquet(config.cache_path("quick/features_train.parquet"))
    features = [c for c in df.columns if c.startswith("q_")]
    gt = io_utils.read_ground_truth()
    gt = gt[gt["s1_id"].isin(set(df["s1_id"].unique()))]
    truth = gt.explode("matches").dropna().rename(columns={"matches": "cand_id"})
    df = df.merge(truth.assign(label=np.int8(1)), on=["s1_id", "cand_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(np.int8)
    n_true = gt.set_index("s1_id")["matches"].str.len()

    hold = _holdout(df["s1_id"])
    _log(f"train rows {(~hold).sum():,}, holdout rows {hold.sum():,}, positives {df['label'].mean():.3f}")
    params = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_child_samples=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=config.N_JOBS, verbose=-1, seed=config.SEED)
    dtr = lgb.Dataset(df.loc[~hold, features], df.loc[~hold, "label"])
    dva = lgb.Dataset(df.loc[hold, features], df.loc[hold, "label"], reference=dtr)
    model = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    _log(f"best iteration {model.best_iteration}, holdout logloss {model.best_score['valid_0']['binary_logloss']:.4f}")

    hv = df.loc[hold, ["s1_id", "cand_id", "label", "q_dense"]].copy()
    hv["prob"] = model.predict(df.loc[hold, features], num_iteration=model.best_iteration)
    hold_s1 = n_true[n_true.index.isin(set(hv["s1_id"]))]
    # S1 entities in the holdout by hash, including any without candidates
    all_hold = gt[_holdout(gt["s1_id"])].set_index("s1_id")["matches"].str.len()
    results = []
    for excl in (False, True):
        for t in np.arange(0.20, 0.91, 0.02):
            results.append((excl, round(float(t), 2), macro_f05(select(hv, t, excl), all_hold)))
    best_excl, best_t, best_f = max(results, key=lambda r: r[2])
    # reference: the dense score alone
    dense_ref = max(((t, macro_f05(select(hv.assign(prob=hv["q_dense"]), t, True), all_hold))
                     for t in np.arange(0.85, 0.99, 0.005)), key=lambda r: r[1])
    _log(f"holdout macro F0.5: model {best_f:.4f} (threshold {best_t}, exclusive {best_excl}); "
         f"dense-score-only baseline {dense_ref[1]:.4f} (threshold {dense_ref[0]:.3f}); "
         f"{len(all_hold):,} holdout S1 ({len(hold_s1):,} with candidates)")

    # final model on the whole sample with the early-stopped number of rounds
    final = lgb.train(params, lgb.Dataset(df[features], df["label"]), model.best_iteration)
    out = config.MODEL_DIR / run_id
    out.mkdir(parents=True, exist_ok=True)
    final.save_model(str(out / "model.txt"))
    imp = pd.Series(final.feature_importance("gain"), index=features).sort_values(ascending=False)
    imp.to_csv(out / "importance.csv")
    meta = {"features": features, "threshold": best_t, "exclusive": best_excl, "holdout_f05": best_f,
            "dense_only_f05": dense_ref[1], "rounds": model.best_iteration, "params": params,
            "grid": results}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    _log(f"saved {out}; top features: {', '.join(imp.index[:8])}; total {time.time() - t0:.0f}s")


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
                                   "prob": model.predict(b[meta["features"]]).astype(np.float32)}))
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
    parser.add_argument("step", choices=["text", "features", "train", "submit"])
    parser.add_argument("--split", choices=["train", "test"])
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.step == "text":
        for split in ([args.split] if args.split else config.SPLITS):
            build_text(split)
    elif args.step == "features":
        build_features(args.split)
    elif args.step == "train":
        train(args.run_id)
    else:
        submit(args.run_id)

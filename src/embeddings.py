"""Dense embeddings: dense blocking pass and embedding-similarity features. Owner: Adithya (AD-2, AD-5). GPU.

Model: intfloat/multilingual-e5-small (MIT); every text is prefixed with "query: " (symmetric matching).
Two views per record: "{tag}" = name + address, "{tag}-name" = name only.
Outputs:
  cache/emb/{tag}_{split}.npy                 float16, L2-normalized, row i = cache/emb/{tag}_{split}_ids.parquet row i
  cache/emb/dense_neighbors_{split}.parquet   s1_id, cand_id, score  (the dense pass, handed to Siva)
  cache/feat_model_{split}.parquet            s1_id, cand_id, fm_emb_cos, fm_emb_name_cos, ...
The fine-tuned model (AD-5) is trained 2-fold by S1 so that fm_* features on train stay out-of-fold;
a model fine-tuned on all of train encodes test.

Usage:
  python -m src.embeddings encode              # both views, both splits
  python -m src.embeddings neighbors --k 20    # dense pass for Siva
  python -m src.embeddings features            # fm_* columns for every candidate file that exists
  python -m src.embeddings all
"""
import argparse
import time
import unicodedata

import numpy as np
import pandas as pd

from . import config, io_utils

# torch is imported only inside _load_model: on Windows, torch and faiss bundle different OpenMP
# runtimes that abort when both load in one process, so the neighbour search below is plain numpy.

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_TAG = "e5s"


def _log(msg: str) -> None:
    print(f"[embeddings {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _basic_clean(text: str) -> str:
    """Fallback cleanup used only while cache/records.parquet (Bhanu, BH-2) does not exist yet."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower().replace("&", " and ")
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def load_records() -> pd.DataFrame:
    """Records with split, source, entity_id, country, name_core, addr_norm."""
    path = config.CACHE_DIR / "records.parquet"
    if path.exists():
        rec = pd.read_parquet(path, columns=["split", "source", "entity_id", "country", "name_core", "addr_norm"])
    else:
        _log("cache/records.parquet not found; using basic cleanup of the raw files until BH-2 lands")
        rec = io_utils.read_all_records()
        rec["name_core"] = rec["business_name"].map(_basic_clean)
        rec["addr_norm"] = rec["business_address"].map(_basic_clean)
        rec = rec[["split", "source", "entity_id", "country", "name_core", "addr_norm"]]
    rec["country"] = rec["country"].str.strip()
    return rec


def _texts(rec: pd.DataFrame, view: str) -> list[str]:
    if view == "name":
        return ("query: " + rec["name_core"]).tolist()
    return ("query: " + rec["name_core"] + " | " + rec["addr_norm"]).tolist()


def _load_model(model_name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model.half()
    _log(f"loaded {model_name} on {device}")
    return model


def encode_records(tag: str = DEFAULT_TAG, model_name: str = DEFAULT_MODEL, batch_size: int = 256) -> None:
    """Encode every record of both splits in two views; saves {tag}_{split}.npy and {tag}-name_{split}.npy."""
    rec = load_records()
    model = _load_model(model_name)
    for view, view_tag, max_len in (("full", tag, 128), ("name", f"{tag}-name", 48)):
        model.max_seq_length = max_len
        for split in config.SPLITS:
            part = rec[rec["split"] == split].reset_index(drop=True)
            t0 = time.time()
            emb = model.encode(_texts(part, view), batch_size=batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)
            np.save(config.cache_path(f"emb/{view_tag}_{split}.npy"), emb.astype(np.float16))
            part[["source", "entity_id"]].to_parquet(config.cache_path(f"emb/{view_tag}_{split}_ids.parquet"), index=False)
            _log(f"{view_tag} {split}: {len(part):,} records, dim {emb.shape[1]}, {time.time() - t0:.1f}s")


def load_embeddings(tag: str, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    emb = np.load(config.cache_path(f"emb/{tag}_{split}.npy")).astype(np.float32)
    ids = pd.read_parquet(config.cache_path(f"emb/{tag}_{split}_ids.parquet"))
    return emb, ids


def _topk(queries: np.ndarray, pool: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact inner-product top-k (vectors are L2-normalized, so this is cosine), sorted best first.

    Queries are processed in chunks so the score block stays around 200 MB.
    """
    k = min(k, len(pool))
    chunk = max(1, int(5e7 // len(pool)))
    all_scores, all_idx = [], []
    for start in range(0, len(queries), chunk):
        scores = queries[start:start + chunk] @ pool.T
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-top, axis=1)
        all_idx.append(np.take_along_axis(idx, order, axis=1))
        all_scores.append(np.take_along_axis(top, order, axis=1))
    return np.vstack(all_scores), np.vstack(all_idx)


def dense_neighbors(split: str, tag: str = DEFAULT_TAG, k: int = 20, k_reverse: int = 5) -> pd.DataFrame:
    """Dense blocking pass: S1->S2 and S1->S3 top-k plus the reverse top-k_reverse, within country.

    Countries come from the data (open set). If a country has fewer than k target records,
    the S1 records of that country search the whole target pool of the split instead.
    """
    emb, ids = load_embeddings(tag, split)
    rec = load_records()
    country = ids.merge(rec[rec["split"] == split][["source", "entity_id", "country"]],
                        on=["source", "entity_id"], how="left")["country"].fillna("").to_numpy()
    source = ids["source"].to_numpy()
    entity = ids["entity_id"].to_numpy()

    s1_mask = source == 1
    if config.SAMPLE_FRAC < 1:
        s1_mask &= np.array([io_utils.in_dev_sample(e) for e in entity])

    parts = []
    for target in (2, 3):
        t_mask = source == target
        for c in np.unique(country[s1_mask]):
            q_idx = np.where(s1_mask & (country == c))[0]
            p_idx = np.where(t_mask & (country == c))[0]
            if len(p_idx) < k:
                p_idx = np.where(t_mask)[0]
            if len(q_idx) == 0 or len(p_idx) == 0:
                continue
            # S1 -> target
            scores, nn = _topk(emb[q_idx], emb[p_idx], k)
            parts.append(pd.DataFrame({"s1_id": np.repeat(entity[q_idx], nn.shape[1]),
                                       "cand_id": entity[p_idx][nn.ravel()], "score": scores.ravel()}))
            # target -> S1 (reverse): catches records whose own best S1 is outside that S1's top-k
            r_idx = np.where(t_mask & (country == c))[0]
            if len(r_idx):
                scores, nn = _topk(emb[r_idx], emb[q_idx], k_reverse)
                parts.append(pd.DataFrame({"s1_id": entity[q_idx][nn.ravel()],
                                           "cand_id": np.repeat(entity[r_idx], nn.shape[1]), "score": scores.ravel()}))

    out = (pd.concat(parts, ignore_index=True)
           .sort_values("score", ascending=False)
           .drop_duplicates(["s1_id", "cand_id"])
           .reset_index(drop=True))
    out["score"] = out["score"].astype(np.float32)
    out.to_parquet(config.cache_path(f"emb/dense_neighbors_{split}.parquet"), index=False)
    _log(f"dense neighbors {split}: {out['s1_id'].nunique():,} S1, {len(out):,} pairs, "
         f"{len(out) / max(out['s1_id'].nunique(), 1):.1f} per S1")
    if split == "train":
        _report_recall(out)
    return out


def _report_recall(pairs: pd.DataFrame) -> None:
    """Share of ground-truth pairs found by the dense pass (train only)."""
    gt = io_utils.read_ground_truth().explode("matches").dropna()
    gt = gt[gt["s1_id"].isin(pairs["s1_id"].unique())]
    if gt.empty:
        return
    found = gt.merge(pairs, left_on=["s1_id", "matches"], right_on=["s1_id", "cand_id"], how="left")["score"].notna()
    by_src = found.groupby(gt["matches"].str[:2].to_numpy()).mean()
    _log(f"dense pass pair recall on train: {found.mean():.4f} "
         + " ".join(f"{s}={r:.4f}" for s, r in by_src.items()))


def _pair_cosine(pairs: pd.DataFrame, tag: str, split: str) -> np.ndarray:
    emb, ids = load_embeddings(tag, split)
    row = pd.Series(np.arange(len(ids)), index=ids["entity_id"].to_numpy())
    a = emb[row.loc[pairs["s1_id"].to_numpy()].to_numpy()]
    b = emb[row.loc[pairs["cand_id"].to_numpy()].to_numpy()]
    return np.einsum("ij,ij->i", a, b).astype(np.float32)


def build_model_features(split: str, tag: str = DEFAULT_TAG) -> pd.DataFrame:
    """fm_emb_cos and fm_emb_name_cos for every row of cache/candidates_{split}.parquet (Siva).

    Keeps any other fm_* columns already in the file (e.g. fm_ce_prob from AD-6).
    """
    cands = pd.read_parquet(config.cache_path(f"candidates_{split}.parquet"), columns=["s1_id", "cand_id"])
    feats = cands.copy()
    feats["fm_emb_cos"] = _pair_cosine(cands, tag, split)
    feats["fm_emb_name_cos"] = _pair_cosine(cands, f"{tag}-name", split)
    path = config.cache_path(f"feat_model_{split}.parquet")
    if path.exists():
        old = pd.read_parquet(path)
        keep = [c for c in old.columns if c.startswith("fm_") and c not in feats.columns]
        if keep:
            feats = feats.merge(old[["s1_id", "cand_id"] + keep], on=["s1_id", "cand_id"], how="left")
    feats.to_parquet(path, index=False)
    _log(f"feat_model {split}: {len(feats):,} rows, columns {[c for c in feats.columns if c.startswith('fm_')]}")
    return feats


def finetune_biencoder(fold: int | None) -> None:
    """MultipleNegativesRankingLoss on S1-S2, S1-S3 and S2-S3 positive pairs plus hard negatives. bf16."""
    raise NotImplementedError("AD-5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["encode", "neighbors", "features", "all"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--k-reverse", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.step in ("encode", "all"):
        encode_records(args.tag, args.model, args.batch_size)
    if args.step in ("neighbors", "all"):
        for split in config.SPLITS:
            dense_neighbors(split, args.tag, args.k, args.k_reverse)
    if args.step in ("features", "all"):
        for split in config.SPLITS:
            if config.cache_path(f"candidates_{split}.parquet").exists():
                build_model_features(split, args.tag)
            else:
                _log(f"skip features for {split}: cache/candidates_{split}.parquet not built yet (SI-3)")

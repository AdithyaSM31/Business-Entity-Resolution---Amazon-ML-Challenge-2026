"""Dense embeddings: dense blocking pass and embedding-similarity features. Owner: Adithya (AD-2, AD-5). GPU.

Model: intfloat/multilingual-e5-small (MIT); every text is prefixed with "query: " (symmetric matching).
It is multilingual, so it can link Devanagari / Kannada names to their Latin spellings, which
character n-gram TF-IDF cannot.
Views: "{tag}" = name + address (default); "{tag}-name" = name only (opt-in with --views full name,
about 9-10 GB of extra disk per split).

Outputs (per split; rows ordered by source, then country, so every (source, country) block is contiguous):
  cache/emb/{tag}_{split}.npy                 float16 memmap, L2-normalized, row i = row i of the ids file
  cache/emb/{tag}_{split}_ids.parquet         source, entity_id, country
  cache/emb/dense_neighbors_{split}.parquet   s1_id, cand_id, score  (the dense pass, handed to Siva)
  cache/feat_model_{split}.parquet            s1_id, cand_id, fm_emb_cos, fm_emb_name_cos, ...
On train, the neighbour search only queries the S1 entities in the dev sample (config.SAMPLE_FRAC);
the S2/S3 pools and every test S1 are always complete.
The fine-tuned model (AD-5) is trained 2-fold by S1 so that fm_* features on train stay out-of-fold;
a model fine-tuned on all of train encodes test.

Usage:
  python -m src.embeddings encode [--views full name]
  python -m src.embeddings neighbors --k 10 --k-reverse 3 --max-per-s1 30
  python -m src.embeddings features
  python -m src.embeddings all
"""
import argparse
import json
import shutil
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import config, io_utils

# torch is imported lazily: on Windows, torch and faiss bundle different OpenMP runtimes that abort
# when both load in one process, so nothing here uses faiss.

DEFAULT_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_TAG = "e5s"
ENCODE_CHUNK = 200_000  # rows encoded and flushed to disk at a time (also the resume granularity)


def _log(msg: str) -> None:
    print(f"[embeddings {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _basic_clean(text: str) -> str:
    """Fallback cleanup used only while cache/records.parquet (Bhanu, BH-2) does not exist yet.

    Accents are stripped only from Latin letters: Devanagari and Kannada vowel signs are marks too
    (Unicode category M*), and dropping them would corrupt those names.
    """
    out = []
    for ch in unicodedata.normalize("NFKD", text.lower().replace("&", " and ")):
        is_mark = unicodedata.category(ch).startswith("M")
        if is_mark and out and out[-1].isascii():
            continue
        out.append(ch if ch.isalnum() or is_mark else " ")
    return " ".join("".join(out).split())


def load_records(split: str) -> pd.DataFrame:
    """One split's records with source, entity_id, country, name_core, addr_norm, sorted by (source, country)."""
    path = config.CACHE_DIR / "records.parquet"
    cols = ["source", "entity_id", "country", "name_core", "addr_norm"]
    if path.exists():
        rec = pd.read_parquet(path, columns=cols, filters=[("split", "==", split)])
    else:
        _log(f"cache/records.parquet not found; basic cleanup of the raw {split} files until BH-2 lands")
        parts = []
        for source in config.SOURCES:
            df = io_utils.read_source(split, source)
            parts.append(pd.DataFrame({
                "source": np.int8(source), "entity_id": df["entity_id"], "country": df["country"],
                "name_core": df["business_name"].map(_basic_clean),
                "addr_norm": df["business_address"].map(_basic_clean),
            }))
            del df
        rec = pd.concat(parts, ignore_index=True)
    rec["country"] = rec["country"].str.strip()
    return rec.sort_values(["source", "country", "entity_id"], kind="stable").reset_index(drop=True)


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


def _rows_done(view_tag: str, split: str) -> int:
    prog_path = config.cache_path(f"emb/{view_tag}_{split}.progress.json")
    emb_path = config.cache_path(f"emb/{view_tag}_{split}.npy")
    return json.loads(prog_path.read_text())["rows_done"] if prog_path.exists() and emb_path.exists() else 0


def _is_complete(view_tag: str, split: str) -> bool:
    ids_path = config.cache_path(f"emb/{view_tag}_{split}_ids.parquet")
    return ids_path.exists() and _rows_done(view_tag, split) >= pq.ParquetFile(ids_path).metadata.num_rows


def encode_records(tag: str = DEFAULT_TAG, model_name: str = DEFAULT_MODEL, views=("full",),
                   batch_size: int = 1024) -> None:
    """Encode every record of both splits; resumable (re-running continues after the last finished chunk).

    Each view costs about 9-10 GB of disk per split (float16, 384 dims), so the name-only view is opt-in.
    """
    model = _load_model(model_name)
    dim = model.get_sentence_embedding_dimension()
    for split in config.SPLITS:
        view_tags = {view: tag if view == "full" else f"{tag}-{view}" for view in views}
        pending = [v for v in views if not _is_complete(view_tags[v], split)]
        if not pending:
            _log(f"{split}: views {list(views)} already complete")
            continue
        rec = load_records(split)
        for view in pending:
            view_tag = view_tags[view]
            model.max_seq_length = 64 if view == "full" else 32
            emb_path = config.cache_path(f"emb/{view_tag}_{split}.npy")
            ids_path = config.cache_path(f"emb/{view_tag}_{split}_ids.parquet")
            prog_path = config.cache_path(f"emb/{view_tag}_{split}.progress.json")
            done = _rows_done(view_tag, split)
            if done == 0:
                need = len(rec) * dim * 2
                free = shutil.disk_usage(emb_path.parent).free
                if need > free - 2e9:
                    raise OSError(f"{view_tag} {split} needs {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB is free "
                                  f"on {emb_path.anchor}; free up space or set BER_CACHE_DIR to another drive")
                rec[["source", "entity_id", "country"]].to_parquet(ids_path, index=False)
                emb = np.lib.format.open_memmap(emb_path, mode="w+", dtype=np.float16, shape=(len(rec), dim))
            else:
                emb = np.load(emb_path, mmap_mode="r+")
            t0 = time.time()
            for start in range(done, len(rec), ENCODE_CHUNK):
                part = rec.iloc[start:start + ENCODE_CHUNK]
                emb[start:start + len(part)] = model.encode(
                    _texts(part, view), batch_size=batch_size, normalize_embeddings=True,
                    convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
                emb.flush()
                prog_path.write_text(json.dumps({"rows_done": start + len(part)}))
                rate = (start + len(part) - done) / (time.time() - t0)
                _log(f"{view_tag} {split}: {start + len(part):,}/{len(rec):,} rows, {rate:,.0f}/s, "
                     f"eta {(len(rec) - start - len(part)) / rate / 60:.0f} min")
            del emb
        del rec


def load_embeddings(tag: str, split: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Memory-mapped float16 embeddings and their ids (source, entity_id, country)."""
    emb = np.load(config.cache_path(f"emb/{tag}_{split}.npy"), mmap_mode="r")
    ids = pd.read_parquet(config.cache_path(f"emb/{tag}_{split}_ids.parquet"))
    return emb, ids


def _topk(queries: np.ndarray, pool: np.ndarray, k: int, q_chunk: int = 1024,
          p_chunk: int = 400_000) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k by inner product (= cosine for normalized vectors), best first. GPU when available."""
    import torch

    k = min(k, len(pool))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pool_t = torch.from_numpy(np.ascontiguousarray(pool)).to(device, dtype)
    out_s = np.empty((len(queries), k), dtype=np.float32)
    out_i = np.empty((len(queries), k), dtype=np.int64)
    with torch.inference_mode():
        for qs in range(0, len(queries), q_chunk):
            q = torch.from_numpy(np.ascontiguousarray(queries[qs:qs + q_chunk])).to(device, dtype)
            best_s = best_i = None
            for ps in range(0, len(pool_t), p_chunk):
                s, i = (q @ pool_t[ps:ps + p_chunk].T).topk(min(k, len(pool_t) - ps), dim=1)
                i += ps
                if best_s is not None:
                    s, j = torch.cat([best_s, s], 1).topk(k, dim=1)
                    i = torch.cat([best_i, i], 1).gather(1, j)
                best_s, best_i = s, i
            out_s[qs:qs + len(q)] = best_s.float().cpu().numpy()
            out_i[qs:qs + len(q)] = best_i.cpu().numpy()
    del pool_t
    return out_s, out_i


def _blocks(ids: pd.DataFrame) -> dict[tuple[int, str], np.ndarray]:
    """Row positions of every (source, country) block; contiguous because the files are sorted."""
    return dict(ids.groupby(["source", "country"]).indices)


def dense_neighbors(split: str, tag: str = DEFAULT_TAG, k: int = 10, k_reverse: int = 3,
                    max_per_s1: int = 30, all_s1: bool = False) -> None:
    """Dense blocking pass: S1->S2 and S1->S3 top-k plus the reverse top-k_reverse, within country.

    Countries come from the data (open set). If a country has fewer than k target records, its S1
    records search the whole target pool of the split. The union is capped at max_per_s1 per S1.

    all_s1=True searches with every train S1 (not just the dev sample) and writes
    dense_neighbors_{split}_all.parquet. Train rows are still modelled on the sample only, but
    competition features (how many S1s want this record, is this S1 its best) must see the same
    full field of S1s on train as on test, or they are skewed about 5x on train.
    """
    emb, ids = load_embeddings(tag, split)
    blocks = _blocks(ids)
    entity = pa.array(ids["entity_id"].to_numpy())

    s1_rows = np.where(ids["source"].to_numpy() == 1)[0]
    if split == "train" and config.SAMPLE_FRAC < 1 and not all_s1:
        keep = np.fromiter((io_utils.in_dev_sample(e) for e in ids["entity_id"].to_numpy()[s1_rows]),
                           dtype=bool, count=len(s1_rows))
        s1_rows = s1_rows[keep]
    s1_country = ids["country"].to_numpy()[s1_rows]

    q_parts, c_parts, s_parts = [], [], []
    t0 = time.time()
    for target in (2, 3):
        all_target = np.where(ids["source"].to_numpy() == target)[0]
        for country in np.unique(s1_country):
            q_rows = s1_rows[s1_country == country]
            p_rows = blocks.get((target, country), np.empty(0, dtype=np.int64))
            if len(p_rows) < k:
                p_rows = all_target
            if len(q_rows) == 0 or len(p_rows) == 0:
                continue
            scores, nn = _topk(emb[q_rows], emb[p_rows], k)
            q_parts.append(np.repeat(q_rows, nn.shape[1]))
            c_parts.append(p_rows[nn.ravel()])
            s_parts.append(scores.ravel())
            # reverse: each target record's closest S1s, which catches matches ranked below k from the S1 side
            r_rows = blocks.get((target, country), np.empty(0, dtype=np.int64))
            if len(r_rows) and k_reverse > 0:
                scores, nn = _topk(emb[r_rows], emb[q_rows], k_reverse)
                q_parts.append(q_rows[nn.ravel()])
                c_parts.append(np.repeat(r_rows, nn.shape[1]))
                s_parts.append(scores.ravel())
            _log(f"{split} S1->S{target} {country}: {len(q_rows):,} queries x {len(p_rows):,} pool, "
                 f"{time.time() - t0:.0f}s elapsed")

    pairs = pd.DataFrame({"q": np.concatenate(q_parts), "c": np.concatenate(c_parts),
                          "score": np.concatenate(s_parts).astype(np.float32)})
    del q_parts, c_parts, s_parts
    pairs = (pairs.sort_values(["q", "score"], ascending=[True, False])
             .drop_duplicates(["q", "c"]))
    pairs = pairs[pairs.groupby("q").cumcount() < max_per_s1]
    table = pa.table({"s1_id": entity.take(pa.array(pairs["q"].to_numpy())),
                      "cand_id": entity.take(pa.array(pairs["c"].to_numpy())),
                      "score": pa.array(pairs["score"].to_numpy())})
    suffix = "_all" if all_s1 and split == "train" else ""
    pq.write_table(table, config.cache_path(f"emb/dense_neighbors_{split}{suffix}.parquet"))
    n_s1 = pairs["q"].nunique()
    _log(f"dense neighbors {split}: {n_s1:,} S1, {len(pairs):,} pairs, {len(pairs) / max(n_s1, 1):.1f} per S1, "
         f"{time.time() - t0:.0f}s")
    if split == "train" and not all_s1:  # the _all file is too big to score in memory, and is not a candidate set
        _report_recall(table.select(["s1_id", "cand_id"]).to_pandas(), set(ids["entity_id"].to_numpy()[s1_rows]))


def _report_recall(pairs: pd.DataFrame, s1_ids: set) -> None:
    """Share of ground-truth pairs found, over the S1 entities that were queried (train only)."""
    gt = io_utils.read_ground_truth()
    gt = gt[gt["s1_id"].isin(s1_ids)].explode("matches").dropna().rename(columns={"matches": "cand_id"})
    found = gt.merge(pairs.assign(hit=True), on=["s1_id", "cand_id"], how="left")["hit"].fillna(False).to_numpy()
    src = gt["cand_id"].str[:2].to_numpy()
    by_src = " ".join(f"{s}={found[src == s].mean():.4f}" for s in ("S2", "S3"))
    _log(f"dense pass pair recall on train: {found.mean():.4f} ({by_src}) over {len(gt):,} true pairs")


def _pair_cosine(pairs: pd.DataFrame, tag: str, split: str, chunk: int = 2_000_000) -> np.ndarray:
    emb, ids = load_embeddings(tag, split)
    index = pd.Index(ids["entity_id"].to_numpy())
    a_rows, b_rows = index.get_indexer(pairs["s1_id"]), index.get_indexer(pairs["cand_id"])
    if (a_rows < 0).any() or (b_rows < 0).any():
        raise KeyError(f"{tag} {split}: some candidate ids have no embedding")
    out = np.empty(len(pairs), dtype=np.float32)
    for s in range(0, len(pairs), chunk):
        a = emb[a_rows[s:s + chunk]].astype(np.float32)
        b = emb[b_rows[s:s + chunk]].astype(np.float32)
        out[s:s + chunk] = np.einsum("ij,ij->i", a, b)
    return out


def build_model_features(split: str, tag: str = DEFAULT_TAG) -> pd.DataFrame:
    """fm_emb_cos and fm_emb_name_cos for every row of cache/candidates_{split}.parquet (Siva).

    Keeps any other fm_* columns already in the file (e.g. fm_ce_prob from AD-6).
    """
    cands = pd.read_parquet(config.cache_path(f"candidates_{split}.parquet"), columns=["s1_id", "cand_id"])
    feats = cands.copy()
    feats["fm_emb_cos"] = _pair_cosine(cands, tag, split)
    if config.cache_path(f"emb/{tag}-name_{split}.npy").exists():
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
    parser.add_argument("--views", nargs="+", default=["full"], choices=["full", "name"],
                        help="'name' adds a name-only view (about 9-10 GB more disk per split)")
    parser.add_argument("--k", type=int, default=10, help="neighbours per S1 per target source")
    parser.add_argument("--k-reverse", type=int, default=3, help="S1 neighbours per S2/S3 record")
    parser.add_argument("--max-per-s1", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--all-s1", action="store_true",
                        help="train only: search with every S1 and write dense_neighbors_train_all.parquet")
    args = parser.parse_args()

    if args.step in ("encode", "all"):
        encode_records(args.tag, args.model, args.views, args.batch_size)
    if args.step in ("neighbors", "all"):
        for split in (["train"] if args.all_s1 else config.SPLITS):
            dense_neighbors(split, args.tag, args.k, args.k_reverse, args.max_per_s1, all_s1=args.all_s1)
    if args.step in ("features", "all"):
        for split in config.SPLITS:
            if config.cache_path(f"candidates_{split}.parquet").exists():
                build_model_features(split, args.tag)
            else:
                _log(f"skip features for {split}: cache/candidates_{split}.parquet not built yet (SI-3)")

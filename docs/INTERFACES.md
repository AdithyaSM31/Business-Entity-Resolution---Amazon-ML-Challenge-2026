# Interfaces between pipeline stages

Every stage reads and writes these files under `cache/`, so the four of us can work in parallel.
**Change this spec only through a PR, and post in the team chat when you do.**

General rules
- Record key is **(split, entity_id)**: IDs may repeat between train and test. Pair key is **(s1_id, cand_id)** inside one split.
- Join on keys, never on row order.
- Text columns: `""` when missing, never NaN. Numeric features: float32, NaN when not computable (LightGBM handles NaN).
- Feature columns carry the owner's prefix, so files never collide: `fn_` / `fa_` (Bhanu), `fc_` / `fc2_` (Siva), `fm_` (Adithya).
- Train-side model outputs (`preds`, `fm_*`, `fc2_*`) must be **out-of-fold**, using `cache/folds.parquet`.

## `cache/records.parquet` (Bhanu, `normalize.py`)
One row per record of all six source files.

| column | type | meaning |
|---|---|---|
| split | str | `train` / `test` |
| source | int8 | 1 / 2 / 3 |
| entity_id | str | as given (`S1-…`, `S2-…`, `S3-…`) |
| country | str | raw label, whitespace-stripped (open set) |
| name_raw, addr_raw | str | untouched input |
| name_norm | str | lowercase, accents stripped, punctuation removed, canonical tokens |
| name_core | str | `name_norm` without legal form and stopwords |
| legal_form | str | canonical class (`INC`, `LLC`, `LTD`, `PVT_LTD`, `SARL`, …) or `""` |
| name_alts | list[str] | extra name variants from a DBA / trading-as split (core form); may be empty |
| name_phon | str | phonetic skeleton of `name_core` |
| addr_norm | str | canonical tokens (st, rd, ave, ste, nr, opp, …) |
| addr_numbers | list[str] | number tokens, separators normalized (`12/3/456` → `12-3-456`) |
| postcode | str | 5–6 digit code if present |
| city, state | str | best guess, `""` if unknown |
| unit | str | suite / floor / shop number |
| is_landmark | bool | address uses near / opp / behind … |
| landmark | str | the landmark phrase (`sbi atm`) |

v0 may fill only `name_norm`, `name_core`, `addr_norm`; the other columns can be `""` / empty until v1.

## `cache/labels.parquet` and `cache/folds.parquet` (Arushi, `folds.py`)
- labels: `s1_id, cand_id, label` (one row per ground-truth pair, label = 1)
- folds: `s1_id, country, n_true, fold` (every training S1, including singletons)

## `cache/emb/…` (Adithya, `embeddings.py`)
- `{tag}_{split}.npy`: float16, L2-normalized, open with `np.load(..., mmap_mode="r")` (train is about 9.6 GB); row *i* is the entity in row *i* of `{tag}_{split}_ids.parquet` (`source, entity_id, country`). Rows are sorted by source, then country, so each (source, country) block is contiguous. Tags: `e5s` (name + address), `e5s-name` (name only).
- `dense_neighbors_{split}.parquet`: `s1_id, cand_id, score` (the dense blocking pass for Siva). Up to 10 neighbours per S1 per target source plus reverse neighbours, capped at 30 per S1. Train covers only the 20% dev sample of S1 (`BER_SAMPLE_FRAC=0.2`); test covers every S1.

## `cache/candidates_{split}.parquet` (Siva, `blocking.py`)
The **final pruned** set: exactly what the model scores and exactly what is written to `candidate_pairs.tsv`.

| column | type | meaning |
|---|---|---|
| s1_id, cand_id | str | pair key |
| cand_source | int8 | 2 or 3 |
| blk_<pass> | bool | found by pass (`tfidf_name`, `tfidf_full`, `rare_token`, `addr_key`, `dense`, `reverse`) |
| bs_<pass> | float32 | that pass's score, NaN if not found |
| prune_score | float32 | score used for top-N pruning |
| prune_rank | int16 | rank within the S1 (0 = best) |

## Feature files (one row per candidate row)
- `feat_pair_{split}.parquet`: Bhanu, `fn_*` and `fa_*`
- `feat_ctx_{split}.parquet`: Siva, `fc_*`
- `feat_ctx2_{split}_{run_id}.parquet`: Siva, `fc2_*` built from stage-1 run `run_id`
- `feat_model_{split}.parquet`: Adithya, `fm_*`

## `cache/preds/{run_id}_{split}.parquet` (Adithya, `train.py` / `predict.py`)
`s1_id, cand_id, prob` (+ `fold` on train). Train probabilities are out-of-fold.
`run_id` = `YYYYMMDD-HHMM_short-description`, and every run gets a row in `docs/experiments.csv`.

## `output/` (Arushi, `io_utils.write_id_lists`)
`matching_results.tsv` and `candidate_pairs.tsv`, following the rules in the problem statement.
Check both with `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` before every upload.

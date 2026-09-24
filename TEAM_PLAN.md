# Team Chai-Square: plan for Amazon ML Challenge 2026

Window: **25 Sep 00:00 IST → 27 Sep 23:59 IST**. 5 leaderboard submissions per day (15 total).
Metric: macro F0.5 per Source 1 entity. A false merge costs more than a miss, and a singleton (an S1 with no matches) scores 1.0 only if we predict nothing.
Data formats between stages: [docs/INTERFACES.md](docs/INTERFACES.md). Experiment log: [docs/experiments.csv](docs/experiments.csv).

## Roles

| Member | Role | Owns (files) |
|---|---|---|
| **Adithya Sankar Menon** (lead) | Models & GPU: embeddings, fine-tuning, LightGBM training, integration | `config.py`, `embeddings.py`, `cross_encoder.py`, `train.py`, `predict.py`, `run_pipeline.py` |
| **B Bhanu Chandra Rekha** | Data checks, text normalization, pair similarity features | `eda.py`, `normalize.py`, `features_pair.py`, `docs/EDA.md` |
| **Rayudu Siva Sai Prakash** | Blocking / candidate generation, context features | `blocking.py`, `features_context.py` |
| **Arushi Ranjan** | Metric & validation, decision layer, submissions, packaging | `metrics.py`, `folds.py`, `postprocess.py`, `io_utils.py` (writers), `scripts/`, `docs/experiments.csv` |

Every file has one owner. If you need a change in someone else's file, ask them or open a PR they review.

## Handoffs (who waits on whom)

| Handoff | From → to | Due (IST) |
|---|---|---|
| EDA answers (a)–(c) | Bhanu → Arushi, Siva | 25 Sep 12:00 |
| `records.parquet` v0 | Bhanu → Siva, Adithya | 25 Sep 12:00 |
| labels, folds, metric | Arushi → everyone | 25 Sep 13:00 |
| `candidates` v0 | Siva → Bhanu, Adithya | 25 Sep 14:00 |
| dense neighbours (pretrained e5) | Adithya → Siva | 25 Sep 16:00 |
| `records.parquet` v1 | Bhanu → everyone | 25 Sep 17:00 |
| `candidates` v1 | Siva → Bhanu, Adithya | 25 Sep 19:00 |
| pair features v1 | Bhanu → Adithya | 25 Sep 21:00 |
| stage-1 out-of-fold predictions | Adithya → Arushi, Siva | 25 Sep 22:00, then after every run |
| error sheets | Arushi → Bhanu, Siva | after every model run |
| fine-tuned dense neighbours, stage-2 features | Adithya ↔ Siva | 26 Sep 15:00 |
| methodology doc sections | everyone → Arushi | 27 Sep 18:00 |

Rule: while a dependency is not ready, build against its v0 or a stub. Nobody waits idle.

## Tasks

### Adithya: Models & GPU (lead)
**Day 1 (25 Sep)**
- [ ] AD-1 Push this scaffold; get everyone cloned and installed before kickoff.
- [ ] AD-2 `embeddings.py`: encode all records with pretrained `intfloat/multilingual-e5-small` (MIT) on the GPU (`"query: {name_core} | {addr_norm}"` plus a name-only variant). Run an exact top-K search within country (S1→S2, S1→S3 and reverse) and hand `dense_neighbors_{split}.parquet` to Siva. Add `fm_emb_cos` features.
- [ ] AD-3 `train.py` / `predict.py`: stage-1 LightGBM on `folds.parquet`, early stopping, out-of-fold predictions, feature importance. First run on blocking scores + whatever pair features exist.
- [ ] AD-4 First real model → submission #2.

**Day 2 (26 Sep)**
- [ ] AD-5 Fine-tune the bi-encoder (MultipleNegativesRankingLoss, bf16, max_seq_length 64, batch 128, 1–2 epochs). Positives: S1–S2, S1–S3 and S2–S3 pairs that share an S1; hard negatives from the candidates. Train 2-fold by S1 so `fm_*` stays out-of-fold; a full-data model encodes test. Re-run the dense pass for Siva.
- [ ] AD-6 Cross-encoder, only if error analysis shows meaning-level mistakes: `microsoft/mdeberta-v3-base` (MIT), bf16, max_len 96, 2-fold → `fm_ce_prob`.
- [ ] AD-7 Stage-2 LightGBM with Siva's `fc2_*` and the `fm_*` features; light tuning.

**Day 3 (27 Sep)**
- [ ] AD-8 Seed ensemble (3–5 seeds × 5 folds). France pseudo-labelling experiment (pairs with p > 0.98 that are mutual best); keep it only if the leaderboard improves.
- [ ] AD-9 `run_pipeline.py` end-to-end from raw data; time each stage for the README; final inference.
- [ ] AD-10 Write the "Model architecture" section of the methodology doc.

### Bhanu: Data checks, normalization, pair features
**Day 1**
- [ ] BH-1 (first 2 h) `eda.py` → `docs/EDA.md`. Post these in the chat as soon as you know them: (a) can an S2/S3 ID match two S1s? (b) do matched pairs always share the country string? (c) singleton rate and match-count distribution.
- [ ] BH-2 `normalize.py` v0 within ~2 h: lowercase, accent strip (`unicodedata`, not `unidecode`), punctuation, `&` → and, legal-form strip → `records.parquet`.
- [ ] BH-3 v1: legal-form class, DBA split, name and address abbreviation maps, address parts (numbers, postcode, city, state, unit, landmark).
- [ ] BH-4 `features_pair.py` v1: rapidfuzz suite (ratio, partial, token_sort, token_set, Jaro-Winkler, Levenshtein) on `name_norm`, `name_core` and `addr_norm`; char-3-gram Jaccard; TF-IDF cosine; postcode / number / city / state match flags. Use multiprocessing.

**Day 2**
- [ ] BH-5 Mine abbreviation pairs from matched training pairs; phonetic skeleton (Shree/Shri/Sri, Laxmi/Lakshmi); French rules from the test France records (SARL/SAS/SA/EURL, rue/bd/av, bis/ter, CEDEX).
- [ ] BH-6 Features v2: highest IDF of a token found in only one name, number conflicts in names, legal-form conflict, acronym match, Monge-Elkan, best score across DBA variants, landmark overlap.
- [ ] BH-7 Work through Arushi's error sheets for name and address mistakes.

**Day 3**
- [ ] BH-8 Freeze normalization and features by 14:00.
- [ ] BH-9 Write the "Normalization" and "Feature engineering" doc sections; reproducibility test on your own laptop.

### Siva: Blocking, context features
**Day 1**
- [ ] SI-1 `blocking.py` v0 on records v0: char 3–5-gram TF-IDF on names, cosine top-K (K ≈ 20) within each country value, S1→S2 and S1→S3, chunked sparse products.
- [ ] SI-2 Blocking report with Arushi's helpers: pair recall, oracle F0.5, candidates per S1, per pass and for the union.
- [ ] SI-3 Remaining passes: name + address TF-IDF, rare token, address key (postcode + street number), reverse direction, Adithya's dense pass. Union, prune to top-N → `candidates_{split}.parquet` v1. Target ≥ 99% pair recall.

**Day 2**
- [ ] SI-4 `features_context.py` (`fc_*`): name and address frequency (chains, malls), near-duplicate count per S1, rank, reverse rank, gap to best, pass flags and scores.
- [ ] SI-5 Stage-2 features (`fc2_*`) from Adithya's out-of-fold stage-1 predictions: rank and margin by probability, mutual best, best competing probability, cross-source support.
- [ ] SI-6 Blocking v2: learned pruning (logistic regression on pass scores); study the ground-truth pairs no pass finds; France candidate counts and score distributions vs US and India.

**Day 3**
- [ ] SI-7 Freeze blocking by 14:00; final test candidates.
- [ ] SI-8 Write the "Candidate generation" doc section (per-pass marginal recall, recall ceiling, reduction ratio); clean-environment rerun from raw data following the README.

### Arushi: Metric, decision layer, submissions, packaging
**Day 1**
- [ ] AR-1 (first 2 h) `folds.py`: `labels.parquet`, `folds.parquet` (5 folds, stratified by country × match-count bucket). `metrics.py`: exact macro F0.5 with tests (PDF example = 0.714) and blocking-report helpers.
- [ ] AR-2 Writers in `io_utils.py` plus a validator wrapper. Baseline: TF-IDF top-1 per source above a tuned threshold → submission #1 (checks the format, gives a leaderboard anchor).
- [ ] AR-3 Set up the experiment log; leave-one-country-out scoring (train US → score India, and the reverse).
- [ ] AR-4 `postprocess.py` v1: global threshold tuned on out-of-fold predictions (optionally one per source), plus exclusivity if BH-1 confirms it.

**Day 2**
- [ ] AR-5 Isotonic calibration; expected-F0.5 set selection (Poisson-binomial DP); optional "has any match" gate. Compare against the plain threshold on out-of-fold and leave-one-country-out scores.
- [ ] AR-6 After each model: error sheets with the 100 worst false positives and false negatives (raw fields) in `docs/errors/` for Bhanu and Siva.
- [ ] AR-7 Run submissions (see below).

**Day 3**
- [ ] AR-8 Final decision settings with Adithya; `scripts/make_submission_zip.py` → `Chai-Square_submission.zip`; pin `requirements.txt`; README run steps.
- [ ] AR-9 Compile `Documentation_template.md` from everyone's sections; final validator run.

## Sync points (IST, 15 minutes each)
Each person says what number they moved (recall, CV F0.5, …) and what is blocking them.

| Day | Times | Milestone |
|---|---|---|
| 25 Sep | 10:00 · 14:00 · 19:00 · 22:30 | 10:00 formats frozen · 14:00 records v0, metric, candidates v0 · 19:00 candidates v1, first out-of-fold score · 22:30 choose submissions |
| 26 Sep | 10:00 · 15:00 · 21:00 | stage 2, fine-tuned embeddings, set selection |
| 27 Sep | 10:00 · 14:00 · 18:00 · 21:00 | 14:00 feature freeze · 18:00 code freeze · 21:00 zip ready (buffer until 23:59) |

## Submissions (5 per day)
- Arushi uploads from one laptop (the rules forbid simultaneous logins); Adithya approves what goes up.
- Every upload: validator PASS → row in `docs/experiments.csv` → upload → public score in the same row.
- One change per submission, so we know what moved the score.
- The final choice favours the best CV + leave-one-country-out result that is also good on the public leaderboard. The public board is only a subset of the test set, and the private board decides the ranking.

| # | Day | Tests |
|---|---|---|
| 1 | 25 Sep | Baseline TF-IDF + threshold (format check, anchor) |
| 2 | 25 Sep | Stage-1 LightGBM + tuned threshold |
| 3 | 25 Sep | Normalization v1 + pair features v1 + exclusivity |
| 4–5 | 25 Sep | Spare |
| 6 | 26 Sep | Fine-tuned dense blocking + embedding features |
| 7 | 26 Sep | Stage 2 (context features) |
| 8 | 26 Sep | Expected-F0.5 set selection |
| 9 | 26 Sep | Cross-encoder feature (if built) |
| 10 | 26 Sep | Spare |
| 11 | 27 Sep | Seed ensemble |
| 12 | 27 Sep | France pseudo-labels |
| 13–14 | 27 Sep | Final candidates |
| 15 | 27 Sep | Spare |

## Git workflow
- One branch per task: `<name>/<topic>` (e.g. `siva/blocking-v1`). Open a PR into `main`; Adithya merges. Merge at least twice a day.
- `config.py` and `docs/INTERFACES.md` change only through a PR, announced in the chat.
- Never commit `dataset/`, `cache/`, `models/` or outputs (they are gitignored). Share large cache files through the team drive.

## Machine notes
- GPU work (embeddings, fine-tuning) runs on Adithya's laptop (RTX 3060, 6 GB). Everything else must run on CPU.
- On slow laptops, set `BER_SAMPLE_FRAC=0.2` to develop on a fixed 20% of S1 entities; Adithya runs the full-data jobs.
- Windows: use `python`, not `python3`, and guard multiprocessing entry points with `if __name__ == "__main__":`.
- Rules: final model MIT/Apache-2.0 and at most 8B parameters; no external data, APIs or geocoding; avoid GPL libraries.

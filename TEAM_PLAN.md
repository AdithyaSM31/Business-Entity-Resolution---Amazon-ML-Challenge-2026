# Team Chai-Square: plan for Amazon ML Challenge 2026

Window: **25 Sep 00:00 IST → 27 Sep 23:59 IST**. 5 leaderboard submissions per day (15 total).
Metric: macro F0.5 per Source 1 entity. A false merge costs more than a miss, and a singleton (an S1 with no matches) scores 1.0 only if we predict nothing.
Data formats between stages: [docs/INTERFACES.md](docs/INTERFACES.md). Experiment log: [docs/experiments.csv](docs/experiments.csv).
Step-by-step playbooks with AI-assistant prompts for Bhanu, Siva and Arushi are in the team's Member Playbooks doc.

## What the real data changed (25 Sep, 12:30 IST)

| Finding | Number | Affects |
|---|---|---|
| Size | train: 2.21M S1, 5.03M S2, 5.29M S3 · test: 1.73M S1, 4.89M S2, 5.08M S3 | everyone |
| Training set | **fixed 20% sample** of train S1 = 440,555 entities, 1.53M true pairs (`BER_SAMPLE_FRAC=0.2`); S2/S3 pools and all of test stay complete | everyone |
| Singletons | 5.6% of S1; most S1 have 2–5 matches, some 8+ | Arushi |
| Exclusivity | no S2/S3 ID belongs to two S1s | Arushi |
| Country | 100% of true pairs share the country string | Siva |
| Same-source duplicates | 81% of S1 have 2+ matches inside one source | Siva, Bhanu |
| Distractors | 26% of S2 and 25% of S3 records match nothing | everyone |
| Empty addresses | 3.4% of S2/S3, none in S1 | Bhanu |
| Scripts | Devanagari names in India S2/S3, Kannada state names: never strip marks from non-Latin letters | Bhanu, Siva |
| Name noise | junk prefixes (`-- `, `<< `), legal words at the front (`LLC Moncada …`), web domains as names, typos | Bhanu |
| Address noise | UPPERCASE, reordered components, `5 bis Rue …` in France | Bhanu |
| Dense pass (AD-2, done) | 95.8% pair recall on the train sample at ~29 candidates per S1; test 43.0M pairs, every S1 covered | Siva |
| Uploads | only the registered team leader can submit on Unstop → **Adithya uploads**, Arushi prepares | Arushi |

Scale rules for everyone: 16 GB laptops, so work per split and per source, read parquet with `columns=`/`filters=`, join on integer codes, process pairs in chunks of ~2M, and use `rapidfuzz.process.cpdist(..., workers=-1)` for pairwise string scores.

## Roles

| Member | Role | Owns (files) |
|---|---|---|
| **Adithya Sankar Menon** (lead) | Models & GPU: embeddings, fine-tuning, LightGBM training, integration, **uploads** | `config.py`, `embeddings.py`, `cross_encoder.py`, `train.py`, `predict.py`, `run_pipeline.py` |
| **B Bhanu Chandra Rekha** | Noise catalogue, text normalization, pair similarity features | `eda.py`, `normalize.py`, `features_pair.py`, `docs/EDA.md` |
| **Rayudu Siva Sai Prakash** | Blocking / candidate generation, context features | `blocking.py`, `features_context.py` |
| **Arushi Ranjan** | Metric & validation, match selection, submission prep, packaging | `metrics.py`, `folds.py`, `postprocess.py`, `evaluate.py`, `io_utils.py` (writers), `scripts/`, `docs/experiments.csv` |

Every file has one owner. If you need a change in someone else's file, ask them or open a PR they review.

## Handoffs (who waits on whom)

| Handoff | From → to | Due (IST) |
|---|---|---|
| Dataset zip + `dense_neighbors_{train,test}.parquet` on the team drive | Adithya → everyone | 25 Sep 13:30 |
| Labels, folds (sample), metric | Arushi → everyone | 25 Sep 15:00 |
| `records.parquet` v0 | Bhanu → everyone | 25 Sep 16:00 |
| Baseline submission #1 (dense pass + threshold) | Arushi → Adithya uploads | 25 Sep 18:00 |
| `candidates` v1 | Siva → Bhanu, Adithya | 25 Sep 20:00 |
| `records.parquet` v1 | Bhanu → everyone | 25 Sep 20:00 |
| Pair features v1 | Bhanu → Adithya | 25 Sep 23:30 |
| Stage-1 model, submission #2 | Adithya | 26 Sep 10:00 |
| Error sheets | Arushi → Bhanu, Siva | after every model run |
| Fine-tuned dense neighbours ↔ stage-2 features | Adithya ↔ Siva | 26 Sep 15:00 |
| Methodology doc sections | everyone → Arushi | 27 Sep 18:00 |

Rule: while a dependency is not ready, build against its v0 or a stub. Nobody waits idle.

## Tasks

### Adithya: Models & GPU (lead)
- [x] AD-1 Scaffold pushed.
- [x] AD-2 Embeddings for all 24.2M records (`multilingual-e5-small`, GPU) and the dense pass: 95.8% pair recall on the train sample.
- [ ] AD-2b `fm_emb_cos` for every candidate once Siva's v1 lands (`python -m src.embeddings features`); optional name-only view (~19 GB, ~1 h).
- [ ] AD-3 `train.py` / `predict.py`: stage-1 LightGBM on the sample's 5 folds, out-of-fold predictions, feature importance, test inference → submission #2 (26 Sep 10:00).
- [ ] AD-5 Fine-tune the embedding model (MultipleNegativesRankingLoss, bf16) on sample positives incl. S2–S3 pairs sharing an S1, 2-fold so `fm_*` stays out-of-fold; target dense recall ≥ 98%.
- [ ] AD-6 Cross-encoder `mdeberta-v3-base` (MIT), only if error analysis shows meaning-level mistakes.
- [ ] AD-7 Stage-2 LightGBM with `fc2_*` and `fm_*`.
- [ ] AD-8 Seed ensemble; France pseudo-label experiment.
- [ ] AD-9 `run_pipeline.py` end to end; AD-10 model section of the doc.
- [ ] Uploads: every submission Arushi prepares; post the public score.

### Bhanu: noise catalogue, normalization, pair features
- [ ] BH-1 Noise catalogue → `docs/EDA.md` (25 Sep 15:00)
- [ ] BH-2 Normalization v0, Indic-safe, all 24.2M records in < 25 min (25 Sep 16:00)
- [ ] BH-3 Normalization v1: every INTERFACES column (25 Sep 20:00)
- [ ] BH-4 Pair features v1: train ~13M pairs < 20 min, test ~45M < 60 min (25 Sep 23:30)
- [ ] BH-5 Mined abbreviations, Devanagari → Latin key (`name_latin`), phonetic key, French rules (26 Sep 13:00)
- [ ] BH-6 Pair features v2: discriminators (26 Sep 17:00)
- [ ] BH-7 Error-sheet fixes (26 Sep, after each run)
- [ ] BH-8 Freeze (27 Sep 14:00) · BH-9 doc sections (27 Sep 18:00)

### Siva: blocking, context features
- [ ] SI-1 Candidates v0 = dense pass + exact-name pass (25 Sep 17:00)
- [ ] SI-2 Blocking report on the sample (25 Sep 17:30)
- [ ] SI-3 Rare-token, address-key, phonetic and Latin-name hash-join passes; prune to ≤ 30 per S1; ≥ 98.5% pair recall (25 Sep 20:00)
- [ ] SI-4 Context features incl. same-source duplicate support (26 Sep 13:00)
- [ ] SI-5 Stage-2 features from out-of-fold predictions (26 Sep 15:00)
- [ ] SI-6 Learned pruning, missed-pair study, France check (26 Sep 21:00)
- [ ] SI-7 Freeze (27 Sep 14:00) · SI-8 doc section + clean rerun (27 Sep 18:00)

### Arushi: metric, selection, submission prep, packaging
- [ ] AR-1 Labels + folds for the 440,555 sampled S1; exact and vectorized macro F0.5 with tests (25 Sep 15:00)
- [ ] AR-2 Writers + validator; dense-pass baseline → submission #1 via Adithya (25 Sep 18:00)
- [ ] AR-3 `evaluate.py`, experiment log, leave-one-country-out scoring (25 Sep 22:00)
- [ ] AR-4 Selection v1: tuned threshold, exclusivity, per-source caps (26 Sep 10:00)
- [ ] AR-5 Calibration + vectorized expected-F0.5 set selection (26 Sep 17:00)
- [ ] AR-6 Error sheets after every run · AR-7 submission prep daily
- [ ] AR-8 Final zip (27 Sep 21:00) · AR-9 methodology doc (27 Sep 21:00)

## Sync points (IST, 15 minutes each)
Each person says what number they moved (recall, CV F0.5, …), what they handed off, and what is blocking them.

| Day | Times | Milestone |
|---|---|---|
| 25 Sep | 13:30 · 18:00 · 23:00 | 13:30 walk through this update · 18:00 submission #1, records v0, candidates v0 · 23:00 candidates v1, features status |
| 26 Sep | 10:00 · 15:00 · 21:00 | submission #2, stage 2, fine-tuned embeddings, set selection |
| 27 Sep | 10:00 · 14:00 · 18:00 · 21:00 | 14:00 feature freeze · 18:00 code freeze · 21:00 zip ready (buffer until 23:59) |

## Submissions (5 per day)
- Only the registered team leader can submit on Unstop: **Arushi builds and validates, Adithya uploads.**
- Every upload: validator PASS → row in `docs/experiments.csv` → upload → public score in the same row → tag `sub-<nn>`.
- One change per submission, so we know what moved the score.
- The final choice favours the best CV + leave-one-country-out result that is also good on the public leaderboard; the private board decides the ranking.

| # | Day | Tests |
|---|---|---|
| 1 | 25 Sep | Dense-pass baseline: score threshold + per-source cap |
| 2–5 | 25 Sep | Spare (only if a fix is ready) |
| 6 | 26 Sep | Stage-1 LightGBM + tuned threshold |
| 7 | 26 Sep | + normalization v1, pair features v1, exclusivity |
| 8 | 26 Sep | Stage 2 (context features) + expected-F0.5 selection |
| 9 | 26 Sep | Fine-tuned dense blocking + embedding features |
| 10 | 26 Sep | Spare |
| 11 | 27 Sep | Seed ensemble |
| 12 | 27 Sep | France pseudo-labels |
| 13–14 | 27 Sep | Final candidates |
| 15 | 27 Sep | Spare |

## Git workflow
- One branch per task: `<name>/<topic>` (e.g. `siva/blocking-v1`). Open a PR into `main`; Adithya merges. Merge at least twice a day.
- `config.py` and `docs/INTERFACES.md` change only through a PR, announced in the chat.
- Never commit `dataset/`, `cache/`, `models/` or outputs (they are gitignored). Share large files through the team drive.

## Machine notes
- GPU work (embeddings, fine-tuning, cross-encoder) runs on Adithya's laptop (RTX 3060, 6 GB). Everything else must run on a CPU laptop with 16 GB RAM.
- Always set `BER_SAMPLE_FRAC=0.2` for train work; test is never sampled.
- Keep at least 25 GB free disk for cache files.
- Windows: use `python`, not `python3`, and guard multiprocessing entry points with `if __name__ == "__main__":`.
- Rules: final model MIT/Apache-2.0 and at most 8B parameters; no external data, APIs or geocoding; no GPL libraries.

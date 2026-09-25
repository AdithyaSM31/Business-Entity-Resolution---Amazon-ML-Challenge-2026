# Work Chart — Who Does What, Inputs → Outputs

*Sources of truth: `docs/Chai-Square Member Playbooks.pdf` (task dues & done-criteria — newer than TEAM_PLAN) · [INTERFACES.md](INTERFACES.md) (file schemas) · `TEAM_PLAN.md` (Adithya's tasks, which the playbooks don't cover).*

---

## A. End-to-end flow: whose output feeds whose input

```
 dataset/*.tsv (raw TSVs, read ONLY through io_utils)
   │
   ├─► BH-2/3 normalize ─────► cache/records.parquet ─────┐
   ├─► AD-2 embeddings ──────► cache/emb/*.npy            │
   │                        └► dense_neighbors.parquet ─┐ │
   ├─► AR-1 folds ──────────► cache/labels.parquet      │ │
   │                        └► cache/folds.parquet      │ │
   │                                                    │ │
   │    ════════════ HANDOFF WAVE ══════════════════════╪═╪════►
   │                                                    │ │
   │  SI-1/3 blocking ◄── records + dense_neighbors ◄──┘ │
   │        │                                            │
   │        └──► cache/candidates_{split}.parquet ──┬────┴─────────────┐
   │                                                 │                 │
   │   BH-4 pair features ◄── records + candidates ──┘                 │
   │        └──► feat_pair_{split}.parquet ──┐                         │
   │                                         ├─► AD-3 stage-1 LightGBM │
   │   SI-4 context features ◄── same ───────┘        │                │
   │        └──► feat_ctx_{split}.parquet ────────────┘                │
   │                                     │                             │
   │                                     └── cache/preds/{run_id}      │
   │                                       (out-of-fold on train)      │
   │                                            │                      │
   │   SI-5 stage-2 features ◄── preds ─────────┤                      │
   │        └──► feat_ctx2_{split}_{run_id} ────┤                      │
   │                                            ▼                      │
   │   AD-7 stage-2 LightGBM ◄── fc2_* + fm_* ──┘                      │
   │        └──► stage-2 preds ────────────────────────────────────┐    │
   │                                                              ▼    │
   │   AR-4/5 selection (threshold · exclusivity · expected-F0.5) │    │
   │        │                                                     │    │
   │        ▼                                                     │    │
   │   output/matching_results.tsv  ◄── AR-3 evaluate.py (CV, loco, experiments.csv)
   │   output/candidate_pairs.tsv ◄──── candidates (SI)           │    │
   │        │                                                     │    │
   │        ├──► AR-2 validate → AR-7 upload ──► public score ─────┘    │
   │        └──► AR-6 error sheets: docs/errors/<run_id>_fp.csv/_fn.csv │
   │                   │                                                │
   │                   ├──► BH-7 fixes in normalize / features_pair ────┘
   │                   └──► SI-6 missed-pair fixes in blocking
   │
   └──► (feedback: one change per submission → next run → next error sheets)
```

---

## B. Per-member work: task · input · output · due · done-when

### Bhanu — normalization & pair features
*Owns: `src/eda.py`, `src/normalize.py`, `src/features_pair.py`, `docs/EDA.md`*

| ID | Task | Input | Output | Due (IST) | Done when |
|----|------|-------|--------|-----------|-----------|
| BH-1 | Data checks | raw TSVs via `io_utils` | `docs/EDA.md` (answers a–h) | 25 Sep **12:00** | (a)–(h) answered with numbers; (a)–(c) in chat |
| BH-2 | Normalization v0 | raw TSVs | `cache/records.parquet` (name_norm, name_core, addr_norm) | 25 Sep **12:00** | all 6 files' records present; row count = sum of file sizes |
| BH-3 | Normalization v1 | records v0 | full INTERFACES columns in `records.parquet` | 25 Sep **17:00** | every column filled; 30 records/country eyeballed |
| BH-4 | Pair features v1 | `records` + `candidates_{split}` | `cache/feat_pair_{split}.parquet` (~40 `fn_`/`fa_` cols) | 25 Sep **21:00** | 1 row per candidate; <15 min on full data |
| BH-5 | Abbrevs, phonetics, French | ground-truth pairs, EDA (g) | `src/data/abbrev_mined.csv`, `name_phon`, French rules | 26 Sep **13:00** | mined table saved; France test records normalize sensibly |
| BH-6 | Pair features v2 | feat_pair | discriminator `fn_`/`fa_` features | 26 Sep **17:00** | CV F0.5 does not drop; new feats in importance |
| BH-7 | Error-sheet fixes | `docs/errors/<run_id>_fp.csv`, `_fn.csv` | fixes in `normalize.py`/`features_pair.py` | after each run | top 3 fixable causes fixed or noted unfixable |
| BH-8 | Freeze | — | — | 27 Sep **14:00** | no normalization/feature changes after |
| BH-9 | Doc sections | code + EDA | "Normalization" + "Feature engineering" sections → Arushi | 27 Sep **18:00** | ≤1,200 words, plain markdown |

### Siva — blocking & context features
*Owns: `src/blocking.py`, `src/features_context.py` — **his recall is our score ceiling: target ≥99% pair recall at ~30–50 candidates/S1**; `candidates_{split}.parquet` = exactly what becomes `candidate_pairs.tsv`*

| ID | Task | Input | Output | Due (IST) | Done when |
|----|------|-------|--------|-----------|-----------|
| SI-1 | Blocking v0 (tfidf_name) | `records` (fallback: lowercased raw names) | `cache/candidates_{split}.parquet` | 25 Sep **14:00** | TF-IDF name pass runs on train+test; candidates v0 saved |
| SI-2 | Blocking report | candidates + ground truth | report table + `docs/blocking_log.csv` | 25 Sep **15:00** | pair recall, oracle F0.5, candidates/S1 printed per pass & country |
| SI-3 | All passes + pruning | records, `dense_neighbors` | candidates **v1** (all INTERFACES columns) | 25 Sep **19:00** | pair recall **≥99%** on train |
| SI-4 | Context features | candidates + records | `cache/feat_ctx_{split}.parquet` (`fc_*`) | 26 Sep **13:00** | 1 row per candidate |
| SI-5 | Stage-2 features | `cache/preds/{run_id}_{split}.parquet` (**OOF only**) | `cache/feat_ctx2_{split}_{run_id}.parquet` (`fc2_*`) | 26 Sep **15:00** | built from out-of-fold preds |
| SI-6 | Pruning v2 + miss study + France | candidates + ground truth | learned prune_score; missed-pair notes; France stats | 26 Sep **21:00** | recall same/better at smaller N; notes posted |
| SI-7 | Freeze | — | final test candidates | 27 Sep **14:00** | no more blocking changes |
| SI-8 | Doc + clean rerun | `src/blocking.py`, `docs/blocking_log.csv` | "Blocking" section → Arushi | 27 Sep **18:00** | pipeline reproduced from raw data, fresh env |

### Arushi — metric, selection, submissions, packaging
*Owns: `src/metrics.py`, `src/folds.py`, `src/postprocess.py`, `src/evaluate.py`, writers in `io_utils.py`, `scripts/`, `docs/experiments.csv`, `docs/errors/`*

| ID | Task | Input | Output | Due (IST) | Done when |
|----|------|-------|--------|-----------|-----------|
| AR-1 | Labels, folds, metric | ground truth | `cache/labels.parquet`, `cache/folds.parquet`, `metrics.py`, `tests/test_metrics.py` | 25 Sep **13:00** | tests pass incl. PDF example (**0.714**) |
| AR-2 | Writers + baseline | records, ground truth (TF-IDF baseline) | `write_id_lists`, `scripts/validate.py`, `scripts/baseline.py`, `output/*.tsv` | 25 Sep **16:00** | validator prints PASS; **submission #1** uploaded + logged |
| AR-3 | Eval + experiment log | `cache/preds/{run_id}_train.parquet` (OOF), folds | `src/evaluate.py` (CV + leave-one-country-out), 1 row in `experiments.csv` | 25 Sep **20:00** | prints CV and loco F0.5 for any predictions file |
| AR-4 | Selection v1 | OOF predictions | `postprocess.select_matches` + `tune` → `models/<run_id>/postprocess.json` | 25 Sep **22:00** | tuned threshold (+ exclusivity) beats 0.5 |
| AR-5 | Calibration + expected-F0.5 | OOF preds (+ loco) | isotonic calibration, expected-F0.5 set selection, optional gate | 26 Sep **17:00** | wins on **both** OOF and leave-one-country-out |
| AR-6 | Error sheets | OOF preds + chosen params | `docs/errors/<run_id>_fp.csv` (100) + `<run_id>_fn.csv` (100) | after every run | posted to team |
| AR-7 | Submissions | `output/matching_results.tsv` | upload + `experiments.csv` row + `git tag sub-<nn>` | daily | validator PASS each upload; score recorded |
| AR-8 | Final zip | outputs, code, docs | `Chai-Square_submission.zip` | 27 Sep **21:00** | rebuilt from clean run; validator PASS |
| AR-9 | Methodology doc | everyone's sections | filled `Documentation_template.md` | 27 Sep **21:00** | headings kept; numbers consistent |

### Adithya — models, GPU, uploads (from TEAM_PLAN; playbooks cover only B/S/A)
*Owns: `config.py`, `embeddings.py`, `cross_encoder.py`, `train.py`, `predict.py`, `run_pipeline.py`, uploads*

| ID | Task | Input | Output | Due (IST) | Status |
|----|------|-------|--------|-----------|--------|
| AD-1 | Scaffold | — | repo structure | 25 Sep | ✅ done |
| AD-2 | Embeddings + dense pass | raw TSVs (24.2M records) | `cache/emb/*.npy`, `dense_neighbors_{split}.parquet` | 25 Sep 13:30 | ✅ done (95.8% recall @ ~29/S1) |
| AD-2b | `fm_emb_cos` features | candidates + embeddings | `feat_model_{split}.parquet` (`fm_*`) | 26 Sep | ❌ |
| AD-3 | Stage-1 `train.py`/`predict.py` | `feat_pair` + `feat_ctx` + `folds` | `cache/preds/{run_id}_{split}.parquet` (OOF) | 26 Sep **10:00** | ❌ |
| AD-5 | Fine-tune embeddings | sample positives | model, dense recall ≥98% | 26 Sep 15:00 | ❌ |
| AD-6 | Cross-encoder (only if errors need it) | — | — | optional | ❌ |
| AD-7 | Stage-2 LightGBM | `fc2_*` + `fm_*` | stage-2 preds | 26 Sep | ❌ |
| AD-8/9 | Seed ensemble · France pseudo-labels · `run_pipeline.py` | all stages | end-to-end run | 27 Sep | ❌ |
| AD-10 | Model doc section | code | section → Arushi | 27 Sep 18:00 | ❌ |
| — | **Uploads** (leader-only on Unstop) | Arushi's validated `matching_results.tsv` | public/private LB score → `experiments.csv` | daily | 0/15 used |

---

## C. Handoff matrix (what moves, which file, who waits)

| # | File that moves | From → To | Due (IST) | Unblocks |
|---|---|---|---|---|
| 1 | `utils/validate_submission.py` + `Documentation_template.md` (from `student_resource/`) | Bhanu commits (once) | 25 Sep morning | validation, methodology doc |
| 2 | `cache/records.parquet` v0 | **Bhanu → Siva** (blocking) | 25 Sep **12:00** | SI-1/3 |
| 3 | data-check answers (a)–(c) | **Bhanu → everyone** (chat) | 25 Sep **12:00** | exclusivity/country assumptions |
| 4 | `cache/labels.parquet` + `folds.parquet` + metric | **Arushi → Adithya** (training) | 25 Sep **13:00** | AD-3 |
| 5 | `dense_neighbors_{split}.parquet` | **Adithya → Siva** (blocking pass 4) | 25 Sep 13:30 | SI-3 dense pass |
| 6 | `cache/candidates_{split}.parquet` | **Siva → Bhanu + Adithya** | 25 Sep **14:00** (v0) / **19:00** (v1) | BH-4, scoring |
| 7 | `cache/feat_pair_{split}.parquet` | **Bhanu → Adithya** | 25 Sep **21:00** | AD-3 |
| 8 | `cache/feat_ctx_{split}.parquet` | **Siva → Adithya** | 26 Sep **13:00** | AD-3 |
| 9 | Baseline `matching_results.tsv` | **Arushi → Adithya uploads** | 25 Sep **16:00** | submission #1 |
| 10 | `cache/preds/{run_id}_train.parquet` (OOF) | **Adithya → Siva + Arushi** | 26 Sep **10:00** | SI-5, AR-3/4 |
| 11 | `docs/errors/<run_id>_fp.csv`/`_fn.csv` | **Arushi → Bhanu + Siva** | after every run | BH-7, SI-6 |
| 12 | Doc sections | **everyone → Arushi** | 27 Sep **18:00** | AR-9 |

---

## D. Sync reporting (15 min, three things each: number moved · handed off · blocked on)

| Member | The number they report |
|---|---|
| Bhanu | Fill rate of parsed fields; which new features rank high in Adithya's importance list |
| Siva | Pair recall, oracle F0.5, mean candidates per S1 (train) |
| Arushi | Best out-of-fold F0.5, leave-one-country-out F0.5, public score of last upload |
| Adithya | Model run + submission status |

**Sync times (IST):** 25 Sep 13:30 · 18:00 · 23:00 — 26 Sep 10:00 · 15:00 · 21:00 — 27 Sep 10:00 · 14:00 · 18:00 · 21:00

---

## E. Rules that shape the flow

- **One task per PR**, title `BH-3: normalization v1` style; description includes the command + printed numbers; tag Adithya as reviewer; merge ≥ 2×/day.
- **Escalate in chat when:** a handoff will miss by >1h (state new time, ship a v0) · you need `INTERFACES.md`/`config.py`/someone else's file changed · a full-data run >30 min or OOM · the assistant suggests external data/GPL/hard-coded countries (answer is no) · a number looks too good (F0.5 > 0.98 ⇒ assume leak).
- **Submissions:** validator PASS → `experiments.csv` row → upload (Arushi prepares, **Adithya uploads**) → public score into the same row → `git tag sub-<nn>`. Never spend the day's last slot without asking at the sync.
- **Never:** read raw TSVs except through `io_utils`, tune on in-fold predictions, put ground truth inside feature files, commit `dataset/`/`cache/`/`models/`.

---

> **Due-time discrepancy note:** TEAM_PLAN.md lists slightly later dues (e.g., BH-1 15:00, AR-1 15:00, AR-4 26 Sep 10:00). This chart uses the **Playbooks PDF** times (25 Sep, by Adithya) — they are newer. Worth reconciling TEAM_PLAN.md in a follow-up edit.

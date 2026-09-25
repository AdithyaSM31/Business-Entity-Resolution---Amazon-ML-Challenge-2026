# Roadmap — Business Entity Resolution, Amazon ML Challenge 2026

**Team Chai-Square** · Window: 25 Sep 00:00 IST → 27 Sep 23:59 IST · Metric: macro F0.5

One-line goal: for every Source 1 business, find its matching records in Source 2 and Source 3.

---

## 0. How we're doing — status board

*Snapshot from `main` + this machine's disk, 26 Sep 2026. Last commit: `a1b7139` (25 Sep 23:39 IST).
Shows work **pushed to the repo** — teammate work still sitting on a local machine would not appear here.*

### Scoreboard

| Metric | Value |
|---|---|
| Pipeline stages fully implemented | **2 / 11** — embeddings + EDA only |
| `src/` files still raising `NotImplementedError` | **12 / 16** |
| Leaderboard submissions used | **0 / 15** (`experiments.csv` is header-only) |
| Day-1 handoffs due (25 Sep 13:30 → 23:30) | **1 / 6** evidenced in repo (dense neighbours only) |
| Missing artefacts | `src/evaluate.py`, `Documentation_template.md`, `utils/validate_submission.py` |
| This clone | no `dataset/`, `cache/`, `models/` on disk — run-time artifacts live on teammates' machines |

### Task-by-task

Legend: ✅ done · 🟡 in progress locally (not pushed) · ❌ not started / stub · ⚠️ risk

| ID | Task | Owner | Due (IST) | Status |
|----|------|-------|-----------|--------|
| AD-1 | Scaffold | Adithya | 25 Sep | ✅ `a53b1fb` |
| AD-2 | Embeddings + dense pass (95.8% recall @ ~29/S1) | Adithya | 25 Sep 13:30 | ✅ `20f0144`, `0e21f40`, `5f2eac3` |
| BH-1 | Noise catalogue → `docs/EDA.md` | Bhanu | 25 Sep 15:00 | ✅ `a1b7139` (530-line report; TEAM_PLAN checkbox still unticked) |
| AR-1 | labels + folds + F0.5 metric | Arushi | 25 Sep 15:00 | ❌ stubs in `folds.py`, `metrics.py` — overdue |
| BH-2/3 | normalize v0/v1 → `records.parquet` | Bhanu | 25 Sep 16:00 / 20:00 | ❌ stub in `normalize.py` — overdue |
| SI-1/2 | candidates v0 + blocking report | Siva | 25 Sep 17:00 / 17:30 | ❌ stub in `blocking.py` — overdue |
| AR-2 | writers + baseline submission #1 | Arushi | 25 Sep 18:00 | ❌ stub in `io_utils.py` — overdue |
| SI-3 | multi-pass blocking, ≥98.5% recall, ≤30/S1 | Siva | 25 Sep 20:00 | ❌ stub — overdue |
| AR-3 | `evaluate.py` + experiment log | Arushi | 25 Sep 22:00 | ❌ file does not exist |
| BH-4 | pair features v1 | Bhanu | 25 Sep 23:30 | ❌ stub in `features_pair.py` — overdue |
| **AD-3** | stage-1 LightGBM → submission #2 | Adithya | **26 Sep 10:00 (today)** | ❌ stubs in `train.py`, `predict.py` — blocked by AR-1 + SI-3 + BH-4 |
| AR-4 | selection v1 (threshold, exclusivity) | Arushi | **26 Sep 10:00 (today)** | ❌ stubs in `postprocess.py` |
| SI-4 | context features `fc_*` | Siva | 26 Sep 13:00 | ❌ stub |
| AD-5 | fine-tuned embeddings (recall ≥98%) | Adithya | 26 Sep 15:00 | ❌ stub at `embeddings.py:310` |
| SI-5 | stage-2 features from OOF preds | Siva | 26 Sep 15:00 | ❌ stub |
| BH-5/6 | abbreviations, `name_latin`, features v2 | Bhanu | 26 Sep 13:00 / 17:00 | ❌ not started |
| AD-7 | stage-2 LightGBM | Adithya | 26 Sep | ❌ not started |
| AR-5 | calibration + expected-F0.5 set selection | Arushi | 26 Sep 17:00 | ❌ not started |
| AD-9 / AR-8 | `run_pipeline.py` / final zip script | Adithya / Arushi | 27 Sep | ❌ stubs |
| DOC | fill `Documentation_template.md` | Arushi | 27 Sep 18:00 | ⚠️ template file not in repo yet |

### Critical path right now (Day 2)

```
overdue Day-1 blockers          today's goal
────────────────────────        ─────────────────────────────────────
AR-1 labels/folds   ─┐
SI-3 candidates     ─┼──► BH-4 pair features ──► AD-3 stage-1 model ──► submission #2
AR-2 writers        ─┘                                   ▲
                                                         │
                          everything above is still ❌ — submission #2 (10:00) is at risk
```

- **Submission #1 (baseline)** was due 25 Sep 18:00 — `experiments.csv` shows 0 uploads.
- **Rule still applies:** nobody waits idle — build against v0/stubs. But v0s themselves are now the blocking items; first order of business is AR-1 + BH-2 + SI-1 landing so the rest can move.

---

## 1. Pipeline at a glance

```
dataset/ TSVs
   │
   ├─► [embeddings] Adithya ──► cache/emb/*.npy, dense_neighbors ──┐
   ├─► [normalize]  Bhanu   ──► cache/records.parquet ───────────┐ │
   ├─► [folds]      Arushi  ──► cache/labels, cache/folds ────┐  │ │
   │                                                          │  │ │
   │        ══════ HANDOFF WAVE (everyone builds on v0) ══════╪══╪═╪═►
   │                                                          │  │ │
   ├─► [blocking]   Siva    ──► cache/candidates ───────────┐ │  │ │
   │                   (records + dense_neighbors) ◄────────┼─┼──┘ │
   │                                                        │ │    │
   ├─► [pair feats] Bhanu   ──► feat_pair  ─┐               │ │    │
   ├─► [ctx feats]  Siva    ──► feat_ctx   ─┼─► [LightGBM stage 1] Adithya
   │                                        │        │ out-of-fold preds
   │        HANDOFF: stage-1 preds ─────────┼────────┘
   │                                        │
   ├─► [ctx2 feats] Siva    ──► feat_ctx2 ──┼─► [LightGBM stage 2] Adithya
   │   (from stage-1 preds)                 │        │ prob per candidate
   │                                        ▼        ▼
   └─► [selection]  Arushi: threshold + exclusivity + expected-F0.5 set
                        │
                        ▼
          output/matching_results.tsv   ← uploaded to leaderboard
          output/candidate_pairs.tsv    ← blocking audit (not scored)
                        │
                        ▼
          validate → experiments.csv → error sheets → fix → next submission
```

**Shape of the work:** parallel for the first wave (4 lanes at once), then a
converging chain with two built-in forks (Bhanu ∥ Siva features; ctx2 ∥ stage-2 prep).

---

## 2. Stage map: owner, inputs, outputs

| # | Stage (module) | Owner | Reads | Writes | Feeds |
|---|---|---|---|---|---|
| 1 | `embeddings.py` | Adithya | raw TSVs | `cache/emb/*.npy`, `dense_neighbors_{split}.parquet` | blocking, stage-2 model |
| 2 | `normalize.py` | Bhanu | raw TSVs | `cache/records.parquet` | everyone |
| 3 | `folds.py` | Arushi | ground_truth, records | `cache/labels.parquet`, `cache/folds.parquet` | training, scoring, blocking audit |
| 4 | `blocking.py` | Siva | records, dense_neighbors | `cache/candidates_{split}.parquet` | pair/ctx features, `candidate_pairs.tsv` |
| 5 | `features_pair.py` | Bhanu | candidates, records | `cache/feat_pair_{split}.parquet` | LightGBM |
| 6 | `features_context.py` | Siva | candidates, records | `cache/feat_ctx_{split}.parquet` | LightGBM |
| 7 | `train.py` (stage 1) | Adithya | feat_pair, feat_ctx, folds | `cache/preds/{run_id}_train.parquet` (out-of-fold) | ctx2 feats, selection |
| 8 | `features_context.py` (stage 2) | Siva | stage-1 preds | `feat_ctx2_{run_id}.parquet` | LightGBM stage 2 |
| 9 | `train.py` (stage 2) | Adithya | all features, `fm_*` | stage-2 preds | selection |
| 10 | `postprocess.py` | Arushi | preds, candidates | `output/*.tsv` | leaderboard |
| 11 | `make_submission_zip.py` | Arushi | outputs, code, doc | `<team>_submission.zip` | final judging |

Column prefixes prevent file collisions: `fn_`/`fa_` Bhanu · `fc_`/`fc2_` Siva · `fm_` Adithya.
Full schema: [INTERFACES.md](INTERFACES.md).

---

## 3. Parallel work lanes (Day 1)

```
Adithya ═ embeddings + dense pass ──────── 95.8% pair recall @ 29/S1 ──►┐
Bhanu   ═ EDA + normalize v0/v1 ────────── 24.2M records ─────────────►┤
Arushi  ═ labels + folds + F0.5 metric ─── 440,555 sample S1 ─────────►┤
Siva    ═ blocking v0 (dense + exact) ──── report, ≤30/S1 prune ───────►┤
                                                                        │
        rule: build against v0/stub while waiting — nobody idles       │
        ~15:00–20:00 handoff wave: stubs swapped for real files ◄──────┘
```

Then the chain: **blocking → features (Bhanu ∥ Siva) → stage 1 → ctx2 ∥ stage-2 prep
→ stage 2 → selection → upload**.

---

## 4. Handoffs (who waits on whom)

| Handoff | From → to | Due (IST) |
|---|---|---|
| Dataset + `dense_neighbors` | Adithya → everyone | 25 Sep 13:30 |
| Labels, folds, metric | Arushi → everyone | 25 Sep 15:00 |
| `records.parquet` v0 | Bhanu → everyone | 25 Sep 16:00 |
| Baseline submission #1 | Arushi → Adithya uploads | 25 Sep 18:00 |
| Candidates v1 | Siva → Bhanu, Adithya | 25 Sep 20:00 |
| Pair features v1 | Bhanu → Adithya | 25 Sep 23:30 |
| Stage-1 model, submission #2 | Adithya | 26 Sep 10:00 |
| Error sheets | Arushi → Bhanu, Siva | after every run |
| Stage-2 features ↔ fine-tuned embeddings | Adithya ↔ Siva | 26 Sep 15:00 |
| Doc sections | everyone → Arushi | 27 Sep 18:00 |

---

## 5. Timeline

| Day | Syncs (IST) | Milestones |
|---|---|---|
| 25 Sep | 13:30 · 18:00 · 23:00 | submission #1 · records v0 · candidates v1 · features status |
| 26 Sep | 10:00 · 15:00 · 21:00 | submission #2 (stage 1) · stage 2 · fine-tuned embeddings · set selection |
| 27 Sep | 10:00 · 14:00 · 18:00 · 21:00 | **14:00 feature freeze** · **18:00 code freeze** · 21:00 zip ready (buffer to 23:59) |

Submissions: 5/day, 15 total — one change per submission, every upload logged in
`experiments.csv`. Only Adithya (registered leader) uploads; Arushi prepares.

---

## 6. Roles

| Member | Owns |
|---|---|
| Adithya (lead) | `config.py`, `embeddings.py`, `cross_encoder.py`, `train.py`, `predict.py`, `run_pipeline.py`, uploads |
| Bhanu | `eda.py`, `normalize.py`, `features_pair.py`, `docs/EDA.md` |
| Siva | `blocking.py`, `features_context.py` |
| Arushi | `metrics.py`, `folds.py`, `postprocess.py`, `evaluate.py`, `io_utils.py`, `scripts/`, `docs/experiments.csv` |

One owner per file — changes to someone else's file go through a PR they review.
`config.py` and `INTERFACES.md` change only via PR + chat announcement.

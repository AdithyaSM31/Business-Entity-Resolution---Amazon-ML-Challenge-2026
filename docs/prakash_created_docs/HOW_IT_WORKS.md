# How Our Pipeline Works — Plain-Language Guide

*Team Chai-Square · companion to [ROADMAP.md](ROADMAP.md) (status) and [INTERFACES.md](INTERFACES.md) (file schemas).*

---

## 1. "Dot pocket" → you mean **dot product**

The dot product is how we compare two businesses as vectors:

```
[1, 2, 3] · [4, 5, 6] = (1×4) + (2×5) + (3×6) = 32
```

Take two vectors, multiply them position-by-position, sum the results → one number
that says *how similar they are*. If both vectors have length 1 (which ours do),
this is exactly **cosine similarity**.

**Where it appears in our flow:**

| Use | What it does |
|---|---|
| **Embeddings** | Every record's name+address becomes a vector like `[0.12, -0.44, …]` via `multilingual-e5-small` |
| **Dense blocking** | Dot-product each S1 vector against all others → keep top ~10 neighbours → this builds `dense_neighbors.parquet` (95.8% pair recall) |
| **Model feature** | `fm_emb_cos` = the dot product of the candidate pair's embeddings, fed to LightGBM |

In one line: **dot product = the arithmetic behind "these two texts mean the same thing."**

---

## 2. The flow: who gives what to whom

```
                          ┌─────────────────────────────────────────────┐
                          │            dataset/ (TSV files)             │
                          │  train_source1/2/3.tsv · ground_truth.tsv   │
                          └──────┬──────────┬──────────┬────────┬───────┘
                                 │          │          │        │
        ┌────────────────────────┤          │          │        ├──────────────────┐
        ▼                        ▼          ▼          ▼        ▼                  ▼
 ┌──────────────┐        ┌────────────┐ ┌─────────┐ ┌──────────────┐      ┌──────────────┐
 │ ADITHYA      │        │ BHANU      │ │ ARUSHI  │ │ ADITHYA      │      │ ARUSHI       │
 │ embeddings.py│        │ normalize.py│ │ folds.py│ │ (uses folds) │      │ metrics.py   │
 └──────┬───────┘        └─────┬──────┘ └────┬────┘ └──────┬───────┘      └──────┬───────┘
        │ writes               │ writes      │ writes      │                     │
        ▼                      ▼             ▼             │                     │
 ┌──────────────────┐   ┌──────────────┐ ┌────────────┐     │              ┌──────────────┐
 │cache/emb/*.npy   │   │cache/records │ │cache/labels│     │              │exact +       │
 │dense_neighbors   │   │  .parquet    │ │cache/folds │     │              │vectorized    │
 │  _{split}.parquet│   │ (24.2M rows) │ │ .parquet   │     │              │F0.5 metric   │
 └───────┬──────────┘   └──────┬───────┘ └─────┬──────┘     │              └──────┬───────┘
         │                     │               │            │                     │
         │    ═════════════════╪═══════════════╪════════════╪══ HANDOFF #1 ═══════╪═
         │                     │   (everyone gets records + folds + metric)        │
         ▼                     ▼               │            │                     │
 ┌───────────────────────────────────────┐      │            │                     │
 │ SIVA  blocking.py                    │      │            │                     │
 │ inputs: records + dense_neighbors     │◄─────┘            │                     │
 └──────────────────┬────────────────────┘                   │                     │
                    │ writes                                 │                     │
                    ▼                                        │                     │
         ┌──────────────────────┐                            │                     │
         │cache/candidates_     │═══ HANDOFF #2 ═══════════►│                     │
         │  {split}.parquet     │   (to Bhanu + Adithya)     │                     │
         └─────┬────────┬───────┘                            │                     │
               │        │                                    │                     │
    ┌──────────┘        └──────────────┐                     │                     │
    ▼                                 ▼                     │                     │
┌────────────────┐          ┌────────────────────┐          │                     │
│ BHANU          │          │ SIVA               │          │                     │
│ features_pair  │          │ features_context   │          │                     │
│ → feat_pair    │          │ → feat_ctx         │          │                     │
└───────┬────────┘          └─────────┬──────────┘          │                     │
        │ HANDOFF #3                  │                     │                     │
        │ (to Adithya)                │                     │                     │
        ▼                             ▼                     ▼                     │
 ┌─────────────────────────────────────────────────────────────┐                  │
 │ ADITHYA  train.py — LightGBM stage 1                        │                  │
 │ inputs: feat_pair + feat_ctx + folds                        │                  │
 │ writes: cache/preds/{run_id}_train.parquet (OUT-OF-FOLD)    │                  │
 └────────────────────────────┬────────────────────────────────┘                  │
                              │ HANDOFF #4 (stage-1 preds)                        │
                              ▼                                                   │
 ┌────────────────────────────────────────┐   ┌───────────────────────────────┐    │
 │ SIVA  feat_ctx2 (stage-2 features      │   │ ADITHYA  stage-2 LightGBM     │    │
 │ built FROM stage-1 predictions)        │──►│ (adds fm_* embedding feats)   │    │
 └────────────────────────────────────────┘   └──────────────┬────────────────┘    │
                                                             │ probs               │
                                                             ▼                     │
                              ┌──────────────────────────────────────────┐         │
                              │ ARUSHI  postprocess.py                   │         │
                              │ threshold + exclusivity + expected-F0.5  │◄────────┘
                              │ set selection                            │ (metric)
                              └───────────────┬──────────────────────────┘
                                              │ writes
                                              ▼
                              ┌───────────────────────────────────────┐
                              │ output/matching_results.tsv  (SCORED)│
                              │ output/candidate_pairs.tsv           │
                              └──────────────────┬────────────────────┘
                                                 │ validate → log to experiments.csv
                                                 ▼
                              ┌───────────────────────────────────────┐
                              │ ADITHYA uploads to Unstop (leader only)│
                              │ public score → back into experiments.csv│
                              └───────────────────────────────────────┘
```

**The five handoffs, in words:**

| # | What moves | From → To |
|---|---|---|
| 1 | `records.parquet` + `labels`/`folds` + metric | Bhanu & Arushi → **everyone** |
| 2 | `candidates_{split}.parquet` | Siva → Bhanu (for pair features) + Adithya (for scoring) |
| 3 | `feat_pair_{split}.parquet` | Bhanu → Adithya (model input) |
| 4 | stage-1 out-of-fold predictions | Adithya → Siva (to build `fc2_*` stage-2 features) |
| 5 | final `.tsv` files | Arushi prepares → Adithya uploads |

---

## 3. Linear or parallel? → **Parallel start, then converging chain**

**It does NOT run like a single straight line.** Three shapes:

### Shape 1 — Day 1: four lanes at once (maximum parallelism)

```
Adithya ═ embeddings + dense pass ──────── 95.8% pair recall @ 29/S1 ──►┐
Bhanu   ═ EDA + normalize v0/v1 ────────── 24.2M records ─────────────►┤
Arushi  ═ labels + folds + F0.5 metric ─── 440,555 sample S1 ─────────►┤
Siva    ═ blocking v0 (dense + exact) ──── report, ≤30/S1 prune ───────►┤
                                                                        │
   RULE: while a dependency isn't ready, build against its v0/stub      │
   — nobody waits idle → ~15:00–20:00 handoff wave swaps stubs ◄───────┘
```

### Shape 2 — after handoffs: a converging chain (mostly linear)

```
records ─┬─► blocking ─► candidates ─┬─► pair features (Bhanu) ─┐
         └─► (dense neighbours)      └─► ctx features (Siva) ───┼─► stage-1 model
                                                                 │
                    ctx2 (Siva) ◄─── out-of-fold preds ──────────┘
                        │
                        ▼
                    stage-2 model ─► selection ─► upload
```

### Shape 3 — two built-in forks (parallelism inside the chain)

1. **Bhanu ∥ Siva features** — pair features and context features are computed *side by side* from the same candidates file; neither waits for the other.
2. **ctx2 ∥ stage-2 prep** — while Siva builds stage-2 features from stage-1 predictions, Adithya can prep the stage-2 training config.

### The feedback loop (not one-way at the end)

```
upload → public score → experiments.csv row → error sheets → Bhanu/Siva fix
       → next submission (ONE change per submission, so we know what moved the score)
```

---

## 4. Work split: inputs → outputs, every stage mapped

| # | Stage (module) | Owner | Reads (input) | Writes (output) | Feeds (downstream) |
|---|---|---|---|---|---|
| 1 | `embeddings.py` | Adithya | raw TSVs | `cache/emb/*.npy`, `dense_neighbors_{split}.parquet` | blocking, stage-2 model |
| 2 | `normalize.py` | Bhanu | raw TSVs | `cache/records.parquet` | everyone |
| 3 | `folds.py` | Arushi | ground_truth + records | `cache/labels.parquet`, `cache/folds.parquet` | training, scoring, blocking audit |
| 4 | `blocking.py` | Siva | records + dense_neighbors | `cache/candidates_{split}.parquet` | pair/ctx features, `candidate_pairs.tsv` |
| 5 | `features_pair.py` | Bhanu | candidates + records | `cache/feat_pair_{split}.parquet` | LightGBM stage 1 |
| 6 | `features_context.py` | Siva | candidates + records | `cache/feat_ctx_{split}.parquet` | LightGBM stage 1 |
| 7 | `train.py` (stage 1) | Adithya | feat_pair + feat_ctx + folds | `cache/preds/{run_id}_train.parquet` (out-of-fold) | ctx2 features, selection |
| 8 | `features_context.py` (stage 2) | Siva | stage-1 preds | `feat_ctx2_{run_id}.parquet` | LightGBM stage 2 |
| 9 | `train.py` (stage 2) | Adithya | all features + `fm_*` | stage-2 preds | selection |
| 10 | `postprocess.py` | Arushi | preds + candidates | `output/matching_results.tsv`, `output/candidate_pairs.tsv` | **leaderboard** |
| 11 | `make_submission_zip.py` | Arushi | outputs + code + doc | `<team>_submission.zip` | **final judging** |

**File-collision rule:** feature columns carry the owner's prefix, so four people
can write feature tables side by side without ever clashing:
`fn_`/`fa_` = Bhanu · `fc_`/`fc2_` = Siva · `fm_` = Adithya · Arushi owns `output/`.

**Train-test hygiene:** all model outputs on train must be **out-of-fold**
(scored by a model that never trained on that row) — otherwise CV scores lie.

---

## 5. One-paragraph summary

Four teammates start **in parallel**, each owning one file, all talking through
`cache/` files defined by [INTERFACES.md](INTERFACES.md). The parallel wave feeds a
**converging chain** — blocking → two feature streams (parallel) → two model stages →
selection — which ends in the two submission TSVs. Scores flow **back** through
`experiments.csv` and error sheets into the next iteration, one change per submission.
The "dot product" is the arithmetic that powers the front of the pipeline (embeddings →
dense blocking) and appears again as a feature inside the model.

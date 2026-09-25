# End to End — From Raw Dataset to Final Submission

*The whole journey in five acts: what arrives → how the team splits it → where it
merges back → how the outputs are assembled → what gets uploaded.*
Companion docs: [HOW_IT_WORKS.md](HOW_IT_WORKS.md) (concepts) · [WORK_CHART.md](WORK_CHART.md) (task tables) · [ROADMAP.md](ROADMAP.md) (status).

---

## Act 1 — What arrives (the input)

Everything starts with one gitignored folder, `dataset/`, dropped into the repo root:

```
dataset/
├── train/
│   ├── train_source1.tsv    2.21M rows   the deduplicated reference source (S1-…)
│   ├── train_source2.tsv    5.03M rows   noisy source (S2-…)
│   ├── train_source3.tsv    5.29M rows   noisy source (S3-…)
│   └── train_ground_truth.tsv  7.64M pairs   which S2/S3 ids match which S1 id
└── test/
    ├── test_source1.tsv     1.73M rows   → every row needs an answer
    ├── test_source2.tsv     4.89M rows
    └── test_source3.tsv     5.08M rows
```

- **Format:** tab-separated; each row = `entity_id · business_name · business_address · country`.
- **No shared identifiers** between sources — we match on noisy names/addresses only.
- **Countries:** US + India in train; test adds **France** (never hard-code countries).
- **Hard rule:** raw TSVs are read *only* through `src/io_utils.py`, and `dataset/` is never committed.

**Mission:** for each of the 1.73M test S1 rows, output the list of S2/S3 ids that are
the same real business (or empty — 5.6% are singletons). Scored by macro F0.5.

---

## Act 2 — How the team splits the work

**Why split at all:** 24.2M records, a 3-day window, four 16 GB laptops. One person
cannot write the whole pipeline in time — so each person *owns files*, and the pieces
talk through `cache/` files with fixed schemas ([INTERFACES.md](INTERFACES.md)).

At 10:00 IST on Day 1, one dataset fans out into **four parallel lanes**:

```
                          dataset/*.tsv
                                 │
       ┌─────────────┬───────────┼───────────┬─────────────┐
       ▼             ▼           ▼           ▼             │
  ┌─────────┐   ┌─────────┐ ┌─────────┐ ┌──────────┐       │
  │ BHANU   │   │ ARUSHI  │ │  SIVA   │ │ ADITHYA  │       │
  │ BH-1/2  │   │ AR-1    │ │ SI-1/2  │ │ AD-2     │       │
  │ EDA +   │   │ labels, │ │ blocking│ │ embed +  │       │
  │ normalize│  │ folds,  │ │ v0      │ │ dense    │       │
  │         │   │ metric  │ │         │ │ pass     │       │
  └────┬────┘   └────┬────┘ └────┬────┘ └────┬─────┘       │
       │             │          │            │             │
       ▼             ▼          ▼            ▼             │
 records.parquet  labels +   candidates    dense_neighbors  │
                  folds      _v0           .parquet         │
       │             │          │            │             │
       └─────────────┴────┬─────┴────────────┘             │
                          ▼                                │
              ══ MERGE 1: handoff wave ════════════════════╝
              (files land ~12:00–14:00; nobody waits —
               everyone codes against v0/stubs meanwhile)
```

**The split in one line each:**

| Member | Lane question | First deliverable (due) |
|---|---|---|
| Bhanu | *How dirty is the data, and how do we clean it?* | `records.parquet` v0 — **25 Sep 12:00** |
| Arushi | *How do we score honestly (labels, folds, metric)?* | `labels` + `folds` + F0.5 tests — **25 Sep 13:00** |
| Siva | *Which pairs are even worth comparing?* | `candidates` v0 — **25 Sep 14:00** |
| Adithya | *How do we vectorize text, and how do we train?* | embeddings + dense pass ✅ done |

---

## Act 3 — Where the pieces merge back

After the handoff wave, the four lanes converge — twice.

### Merge 2: candidates + records → two feature streams (still parallel)

```
records.parquet ─────────────┬──► BH-4 pair features ──► feat_pair  (fn_*, fa_*)
                             │
candidates.parquet ──┬───────┴──► SI-4 context feats ──► feat_ctx   (fc_*)
                     │
dense_neighbors ─────┘
```

Both feature files have **one row per candidate pair**, keyed by `(s1_id, cand_id)` —
that shared key is what makes two independently-built tables joinable.

### Merge 3: features → one training matrix → one model

```
feat_pair ──┐
feat_ctx  ──┼── join on (s1_id, cand_id) ──► LightGBM stage 1 (Adithya)
folds     ──┘                                    │
                                                 ▼ out-of-fold probs
                                                 │
                     ┌───────────────────────────┤
                     ▼                           ▼
        SI-5: feat_ctx2 (fc2_* from OOF)   AR-4/5: threshold tuning (OOF)
                     │
                     ▼
        LightGBM stage 2 (+ fm_* embedding feats)
                     │
                     ▼ stage-2 probs per candidate ──► Arushi's selection
```

**Merge rule:** nobody ever joins on row order — always on keys. And model outputs on
train must be **out-of-fold** (scored by a model that never saw that row) so CV is honest.

---

## Act 4 — Putting the outputs together

Two files are assembled from three inputs:

```
 candidates_{test}.parquet (Siva's sieve) ─────────────► candidate_pairs.tsv
                                                               │
 stage-2 probs ──► selection ──► one row per test S1 ──► matching_results.tsv
   (threshold · exclusivity · expected-F0.5 set choice)
```

**Assembly steps (the AR-7 checklist):**

1. **Build** — `run_pipeline.py` (or predict + postprocess for a single run) writes
   both TSVs via `io_utils.write_id_lists`: rows in the test `source1` file order,
   comma-joined ids, `""` for singletons, UTF-8, tab-separated.
2. **Validate** — `python scripts/validate.py` must print **PASS**: every S1 present,
   no duplicate ids, only real S2-/S3- ids, matches ⊆ candidates. *A rejected upload
   costs a submission slot — never skip this.*
3. **Log** — add a row to `docs/experiments.csv` **before** uploading: what changed,
   CV F0.5, leave-one-country-out F0.5.
4. **Upload** — Arushi prepares, **Adithya uploads** (leader-only on Unstop). Record
   the public score in the same `experiments.csv` row; tag the commit `git tag sub-<nn>`.
5. **Learn** — Arushi writes `docs/errors/<run_id>_fp.csv` / `_fn.csv`; Bhanu fixes
   normalization/features, Siva fixes blocking → next run, next submission.
   **One change per submission**, so the score movement is attributable.

### Closing the loop (what happens between submissions)

```
submission ─► public score ─► experiments.csv ─► error sheets
                                                     │
                    BH-7 fixes (normalize/features) ◄┤
                    SI-6 fixes (blocking)           ◄┘
                             │
                             ▼
                       next run → next submission
```

### The final deliverable (27 Sep 21:00, buffer to 23:59)

`scripts/make_submission_zip.py` builds one zip for the judges:

```
Chai-Square_submission.zip
├── output/
│   ├── matching_results.tsv      ← the scored file (same as leaderboard upload)
│   └── candidate_pairs.tsv       ← the sieve, for blocking audit
├── code/business_entity_resolution/
│   ├── src/  scripts/  README.md  requirements.txt  docs/INTERFACES.md
│   └── (reproduce both TSVs from raw data using only this folder)
└── Documentation_template.md     ← the filled methodology write-up (AR-9)
```

---

## The whole story in ten steps

1. **Raw TSVs** land in `dataset/` — 24.2M rows, no shared ids, 3 countries.
2. **Four lanes start in parallel** — normalize, label/score, block, embed.
3. **Handoff wave** — `records`, `labels/folds`, `dense_neighbors` circulate; lanes swap v0s for real files.
4. **Blocking** collapses 17 trillion possible pairs → ~43M candidates (≥99% recall target).
5. **Two feature streams run side by side** — pair features (Bhanu) + context features (Siva), joined on `(s1_id, cand_id)`.
6. **Stage-1 LightGBM** scores candidates → out-of-fold probabilities.
7. **Stage-2** adds features built *from* those probabilities (`fc2_*`, `fm_*`) and retrains.
8. **Selection** turns probabilities into decisions — threshold, exclusivity, expected-F0.5 — producing `matching_results.tsv` + `candidate_pairs.tsv`.
9. **Validate → log → upload → error sheets → fix**, one change per submission, 15 shots total.
10. **Final zip** ships the two TSVs, the runnable code, and the methodology doc — validators pass, judges reproduce, ranks are decided on the private leaderboard.

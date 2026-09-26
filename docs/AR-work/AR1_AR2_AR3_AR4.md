# AR-1 & AR-2 — Work Done So Far

## AR-1: Metrics & Folds

### What we did
- Implemented `src/metrics.py`
  - `f05_entity()` → calculates F0.5 for one business.
  - `macro_f05()` → averages F0.5 across all Source-1 businesses.
  - `blocking_report()` → checks whether true matches are present in generated candidates.
- Implemented `src/folds.py`
  - Creates `labels.parquet` containing known matching pairs.
  - Creates `folds.parquet` for 5-fold validation.
  - Folds are stratified using country + number of true matches.
- Created `tests/test_metrics.py`.

### Testing
```powershell
python -m pytest tests/test_metrics.py -v
```

## AR-2: Baseline & Submission Utilities

### What is AR-2?

AR-2 is about building our **first simple matching system** and creating valid submission files.

### What we did

- **`write_id_lists()`**
  - Converts our predictions into the required TSV format.
  - Makes sure candidate IDs are valid `S2-` or `S3-` IDs.

- **`validate.py`**
  - Runs the official submission validator.
  - Helps catch formatting or invalid-ID errors.

- **`baseline.py`**
  - Combines business **name + address**.
  - Converts the text into **TF-IDF character features**.
  - Uses **cosine similarity** to find similar businesses.
  - Only compares businesses from the **same country**.
  - Finds the top matching candidates from Source 2 and Source 3.
  - Tests different similarity thresholds using **Macro F0.5**.
  - Creates:
    - `matching_results.tsv` → final predicted matches
    - `candidate_pairs.tsv` → top candidate matches

### Why are we doing this?

We need a **simple baseline first** before trying more advanced models.

It gives us a working system that we can later compare our improved approaches against.

### Testing with small data

The dataset is very large, so for quick development testing we used:

```powershell
$env:BER_SAMPLE_FRAC="0.0001"
```


## AR-3: Evaluation & Experiment Logging

### What we did

- Implemented `src/evaluate.py`.
- Loads out-of-fold predictions from `cache/preds/<run_id>_train.parquet`.
- Loads ground truth and validation folds.
- Applies the current match-selection parameters.
- Reports Macro F0.5:
  - Overall
  - Per fold
  - Per country
  - Per `n_true` bucket (`0, 1, 2, 3+`)
  - Per source (S2/S3)
- Reports singleton accuracy and non-singleton empty rate.
- Reports leave-one-country-out (LOCO) F0.5 when predictions are available.
- Appends experiment results to `docs/experiments.csv`.

### Run

```powershell
python -m src.evaluate --run-id <run_id>
```


## AR-4: Selection v1

### What we did

- Implemented `src/postprocess.py`.
- Added `select_matches()` to convert pair probabilities into final match lists.
- Supports:
  - Global probability threshold.
  - Separate thresholds for Source 2 and Source 3.
  - Exclusivity, where a candidate can be assigned to only one Source-1 entity.
  - Tie handling by dropping tied candidates.
- Implemented `tune()` to search thresholds from **0.05 to 0.95** and test:
  - Global vs. per-source thresholds.
  - Exclusivity on/off.
- Uses **Macro F0.5** on out-of-fold predictions to select the best parameters.
- Prints the F0.5 curve to observe how the score changes with the threshold.
- Saves the selected post-processing parameters to `models/<run_id>/postprocess.json`.

### Why?

AR-4 converts model probabilities into the final set of predicted entity matches and tunes the selection threshold to improve Macro F0.5.
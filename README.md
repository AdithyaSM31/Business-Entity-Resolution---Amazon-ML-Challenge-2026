# Business Entity Resolution (Amazon ML Challenge 2026, Team Chai-Square)

Finds which Source 2 and Source 3 business records refer to each Source 1 entity.
Pipeline: normalize → multi-pass blocking → pair features → LightGBM → F0.5-aware match selection.

- Who does what: [TEAM_PLAN.md](TEAM_PLAN.md)
- File formats between stages: [docs/INTERFACES.md](docs/INTERFACES.md)
- Experiment log: [docs/experiments.csv](docs/experiments.csv)

## Setup

1. Clone this repo and copy the challenge `dataset/` folder (with `train/` and `test/`) into its root. `dataset/` is gitignored.
2. Install dependencies (Python 3.11):
   ```
   pip install -r requirements.txt
   ```

## Run

Stages, in order (commands are filled in as each stage lands):

```
python -m src.normalize
python -m src.folds
python -m src.embeddings
python -m src.blocking --split train
python -m src.blocking --split test
python -m src.features_pair --split train
python -m src.features_context --split train
python -m src.train --run-id <run_id>
python -m src.predict --run-id <run_id>
python -m src.run_pipeline
```

Environment variables: `BER_DATA_DIR`, `BER_CACHE_DIR`, `BER_MODEL_DIR`, `BER_OUTPUT_DIR` override the default folders; `BER_SAMPLE_FRAC=0.2` develops on a fixed 20% of Source 1 entities.

## Layout

```
src/        pipeline code (one owner per file, see TEAM_PLAN.md)
scripts/    submission packaging
docs/       interfaces, EDA notes, experiment log, error sheets
cache/      intermediate files (gitignored)
models/     trained models (gitignored)
output/     matching_results.tsv, candidate_pairs.tsv (gitignored)
```

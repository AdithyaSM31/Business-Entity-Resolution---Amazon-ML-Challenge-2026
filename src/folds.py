"""Labels and validation folds over training Source 1 entities. Owner: Arushi (AR-1).

Outputs:
  cache/labels.parquet  s1_id, cand_id, label (=1 for every ground-truth pair)
  cache/folds.parquet   s1_id, country, n_true, fold (0..N_FOLDS-1), stratified by country x match-count bucket
S2/S3 records are never split: every fold is scored against the full S2/S3 pool, like the test set.
Leave-one-country-out (LOCO) validation uses the `country` column directly.
"""


def make_labels_and_folds() -> None:
    raise NotImplementedError("AR-1")


if __name__ == "__main__":
    make_labels_and_folds()

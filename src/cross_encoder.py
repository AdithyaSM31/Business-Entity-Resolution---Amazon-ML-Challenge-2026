"""Optional cross-encoder pair scorer. Owner: Adithya (AD-6). GPU.

Build it only if error analysis shows meaning-level mistakes that string features miss.
Model: microsoft/mdeberta-v3-base (MIT); fall back to intfloat/multilingual-e5-small if too slow.
Input text: "{name} | {address}" for each side; bf16, max_len 96. Train 2-fold by S1 on all positives
plus about 5 hard negatives each, so the fm_ce_prob feature is out-of-fold on train.
Output: column fm_ce_prob added to cache/feat_model_{split}.parquet
"""


def train_and_score(n_folds: int = 2) -> None:
    raise NotImplementedError("AD-6")


if __name__ == "__main__":
    train_and_score()

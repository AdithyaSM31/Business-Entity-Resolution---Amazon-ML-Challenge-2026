"""Dense embeddings: dense blocking pass and embedding-similarity features. Owner: Adithya (AD-2, AD-5). GPU.

Model: intfloat/multilingual-e5-small (MIT); prefix every text with "query: ".
Outputs:
  cache/emb/{tag}_{split}.npy                 float16, L2-normalized, row i = cache/emb/{tag}_{split}_ids.parquet row i
  cache/emb/dense_neighbors_{split}.parquet   s1_id, cand_id, score  (the dense pass, handed to Siva)
  cache/feat_model_{split}.parquet            s1_id, cand_id, fm_emb_cos, fm_emb_name_cos, ...
The fine-tuned model (AD-5) is trained 2-fold by S1 so that fm_* features on train stay out-of-fold;
a model fine-tuned on all of train encodes test.
"""
import argparse


def encode_records(tag: str, model_name: str) -> None:
    raise NotImplementedError("AD-2")


def dense_neighbors(split: str, tag: str, k: int) -> None:
    raise NotImplementedError("AD-2")


def finetune_biencoder(fold: int | None) -> None:
    """MultipleNegativesRankingLoss on S1-S2, S1-S3 and S2-S3 positive pairs plus hard negatives. bf16."""
    raise NotImplementedError("AD-5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--tag", default="e5s")
    args = parser.parse_args()
    encode_records(args.tag, args.model)

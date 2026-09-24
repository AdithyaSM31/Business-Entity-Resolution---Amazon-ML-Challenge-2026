"""Phase-0 data checks. Owner: Bhanu (BH-1). Writes the answers to docs/EDA.md.

Post answers (a)-(c) in the team chat the moment you have them; other tasks depend on them.
  (a) Does any S2/S3 ID appear under two or more S1 entities?        -> Arushi: exclusivity rule
  (b) Do matched pairs always carry the same country string?          -> Siva: block within country
  (c) Singleton rate and distribution of match counts per S1 (0/1/2/3+)
  (d) Can two records from the same source match one S1?
  (e) File sizes per split, source and country
  (f) 50 matched pairs each from S2 and S3: which fields are degraded, and how?
  (g) Test France records: legal forms, street types, postcode format (inputs only)
  (h) Do matched IDs or row order correlate? Check only; never use it as a feature.
"""


def run_checks() -> None:
    raise NotImplementedError("BH-1")


if __name__ == "__main__":
    run_checks()

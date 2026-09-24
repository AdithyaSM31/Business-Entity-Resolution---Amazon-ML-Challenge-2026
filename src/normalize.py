"""Name and address normalization. Owner: Bhanu (BH-2, BH-3, BH-5).

Output: cache/records.parquet, one row per record of all six source files (schema: docs/INTERFACES.md).
Ship v0 (lowercase, accent strip, punctuation, legal-form strip) early so blocking and features can start.

Rules:
- Never branch on country. Map words to one short canonical form instead (street->st, saint->st, road->rd),
  which keeps French "St" (Saint) and English "St" (Street) consistent without knowing the country.
- Strip accents with the stdlib `unicodedata` (NFKD). Do not use `unidecode` (GPL).
"""
import pandas as pd


def normalize_name(name: str) -> dict:
    """Return name_norm, name_core, legal_form, name_alts, name_phon for one business name."""
    raise NotImplementedError("BH-2")


def normalize_address(address: str) -> dict:
    """Return addr_norm, addr_numbers, postcode, city, state, unit, is_landmark, landmark for one address."""
    raise NotImplementedError("BH-3")


def build_records() -> pd.DataFrame:
    """Normalize every record from io_utils.read_all_records() and save cache/records.parquet."""
    raise NotImplementedError("BH-2")


if __name__ == "__main__":
    build_records()

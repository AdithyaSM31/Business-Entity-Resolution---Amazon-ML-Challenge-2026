"""Name and address normalization. Owner: Bhanu (BH-2, BH-3, BH-5).

Output: cache/records.parquet, one row per record of all six source files (schema: docs/INTERFACES.md).
Ship v0 (lowercase, accent strip, punctuation, legal-form strip) early so blocking and features can start.

Rules:
- Never branch on country. Map words to one short canonical form instead (street->st, saint->st, road->rd),
  which keeps French "St" (Saint) and English "St" (Street) consistent without knowing the country.
- Strip accents with the stdlib `unicodedata` (NFKD). Do not use `unidecode` (GPL).

Shared text primitives (BH-1 measures them, BH-2 applies them):
- `LATIN_FOLD` folds Latin letters with their accents, built from unicodedata NFKD but applied *only*
  to Latin code points. A blanket NFKD + drop-combining-marks pass would delete the matras of
  Devanagari, Kannada, Malayalam and Tamil, which is why the fold table is restricted to Latin.
- `dominant_script` / `non_latin_scripts` report which script a string is written in, because S1 is
  100% Latin while 3-6% of S2/S3 names are Devanagari and another 8 scripts are present. Anything
  Latin-only (folding, punctuation removal, token comparisons) must not be applied to the others.
"""
import unicodedata

import pandas as pd

# Code point ranges per script, in priority order: the first range that contains a code point wins.
# Order matters where blocks touch (Hangul Jamo shares code points with the Brahmic blocks).
SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("LATIN", 0x0041, 0x005A), ("LATIN", 0x0061, 0x007A), ("LATIN", 0x00C0, 0x024F),
    ("LATIN", 0x1E00, 0x1EFF),
    ("DEVANAGARI", 0x0900, 0x097F), ("BENGALI", 0x0980, 0x09FF), ("GURMUKHI", 0x0A00, 0x0A7F),
    ("GUJARATI", 0x0A80, 0x0AFF), ("ORIYA", 0x0B00, 0x0B7F), ("TAMIL", 0x0B80, 0x0BFF),
    ("TELUGU", 0x0C00, 0x0C7F), ("KANNADA", 0x0C80, 0x0CFF), ("MALAYALAM", 0x0D00, 0x0D7F),
    ("SINHALA", 0x0D80, 0x0DFF), ("THAI", 0x0E00, 0x0E7F), ("LAO", 0x0E80, 0x0EFF),
    ("TIBETAN", 0x0F00, 0x0FFF), ("MYANMAR", 0x1000, 0x109F), ("KHMER", 0x1780, 0x17FF),
    ("HANGUL", 0xAC00, 0xD7AF),
    ("GREEK", 0x0370, 0x03FF), ("CYRILLIC", 0x0400, 0x04FF), ("ARMENIAN", 0x0530, 0x058F),
    ("HEBREW", 0x0590, 0x05FF), ("ARABIC", 0x0600, 0x06FF), ("SYRIAC", 0x0700, 0x074F),
    ("THAANA", 0x0780, 0x07BF), ("GEORGIAN", 0x10A0, 0x10FF), ("ETHIOPIC", 0x1200, 0x137F),
    ("CHEROKEE", 0x13A0, 0x13FF), ("MONGOLIAN", 0x1800, 0x18AF), ("TIFINAGH", 0x2D30, 0x2D7F),
    ("CJK", 0x2E80, 0x9FFF), ("HIRAGANA", 0x3040, 0x309F), ("KATAKANA", 0x30A0, 0x30FF),
    ("FULLWIDTH", 0xFF00, 0xFFEF),
)
SCRIPT_NAMES: tuple[str, ...] = tuple(dict.fromkeys(name for name, _, _ in SCRIPT_RANGES))
# Private-use code points as the one-letter codes, so a code can never collide with a real character.
_CODE_OF = {name: chr(0xE000 + i) for i, name in enumerate(SCRIPT_NAMES)}
SCRIPT_BY_CODE = {code: name for name, code in _CODE_OF.items()}


class _ScriptTable(dict):
    """Total mapping: a code point in no known block is deleted rather than passed through.

    The real data contains zero-width characters (U+200C among others), which belong to no script and
    would otherwise reach SCRIPT_BY_CODE and raise. Deleting them is also the correct reading here:
    they are invisible, so they must not count towards any script.
    """

    def __missing__(self, cp: int) -> None:
        return None


def _build_script_table() -> _ScriptTable:
    """Code point -> one-letter script code, for a C-speed str.translate().

    Mapped to None, i.e. deleted by translate(), are the characters that belong to no script: ASCII
    digits and punctuation, the Latin-1 punctuation block, and DEL. Deleting them means both callers
    see only script codes. Non-ASCII digits are *not* deleted - they stay, and they count towards
    their own block, so a Malayalam-digit PIN reads as MALAYALAM.
    """
    table = _ScriptTable()
    for name, lo, hi in SCRIPT_RANGES:
        code = _CODE_OF[name]
        for cp in range(lo, hi + 1):
            table.setdefault(cp, code)
    for lo, hi in ((0x0000, 0x0040), (0x005B, 0x0060), (0x007B, 0x00BF)):
        for cp in range(lo, hi + 1):
            table.setdefault(cp, None)
    return table


SCRIPT_TABLE = _build_script_table()


def _build_latin_fold() -> dict[int, str]:
    """Latin code point -> accent-stripped form, derived from unicodedata NFKD.

    Only Latin ranges are folded. Every other script is left byte-identical, which is the whole
    point: NFKD + drop-combining-marks on 'सन ट्रेडिंग' would leave 'सन ट्रेडिग'.
    """
    fold: dict[int, str] = {}
    for lo, hi in ((0x00C0, 0x024F), (0x1E00, 0x1EFF)):
        for cp in range(lo, hi + 1):
            if not unicodedata.category(chr(cp)).startswith("L"):
                continue
            plain = "".join(c for c in unicodedata.normalize("NFKD", chr(cp))
                            if not unicodedata.category(c).startswith("M"))
            if plain and plain != chr(cp):
                fold[cp] = plain
    return fold


LATIN_FOLD = _build_latin_fold()


def fold_latin(text: str) -> str:
    """Lowercase and strip Latin accents. Characters of other scripts pass through untouched."""
    return text.translate(LATIN_FOLD).lower() if not text.isascii() else text.lower()


def dominant_script(text: str) -> str:
    """Name of the script most of the letters of `text` belong to, "LATIN" for pure ASCII.

    Digits count towards their own block's script, so a Malayalam-digit PIN reads as MALAYALAM.
    "EMPTY" is separated from "OTHER" because the empty address is a top-3 noise pattern and would
    otherwise hide inside an uninformative OTHER row; "OTHER" is digits and punctuation only.
    """
    if not text.strip():
        return "EMPTY"
    if text.isascii():
        return "LATIN" if any(ch.isalpha() for ch in text) else "OTHER"
    counts: dict[str, int] = {}
    for code in text.translate(SCRIPT_TABLE):
        counts[code] = counts.get(code, 0) + 1
    if not counts:
        return "OTHER"
    return SCRIPT_BY_CODE[max(counts, key=counts.get)]


def non_latin_scripts(text: str) -> tuple[str, ...]:
    """Every non-Latin script present, sorted. Empty for ASCII and for accented Latin.

    Reported next to the dominant script because a mixed string is the common case here:
    'XVIII/C-37, FIRST FLOOR, GURUVAYOOR ROAD, THRISSUR, കേരളം' is dominant-LATIN but carries
    MALAYALAM, and 36k such test addresses would be invisible in a dominant-script count alone.
    """
    if text.isascii():
        return ()
    found = {SCRIPT_BY_CODE[code] for code in text.translate(SCRIPT_TABLE)}
    found.discard("LATIN")
    return tuple(sorted(found))


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

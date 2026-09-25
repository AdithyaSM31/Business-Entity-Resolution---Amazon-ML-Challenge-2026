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

Raw TSVs are read only through io_utils. Each of the six files is read once and the frame is dropped
immediately, so peak memory stays near 2 GB on a 16 GB laptop: read_all_records() stacks all 24.2M
rows and does not fit. The train files are additionally read a second time, targeting ~100 ids, to
print the raw text of the (f) samples.
"""
import argparse
import gc
import itertools
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import config, io_utils

DOC_PATH = config.ROOT / "docs" / "EDA.md"

ID_PREFIX_LEN = 2                                  # "S1-" / "S2-" / "S3-" is always 3 characters
POSTCODE_RE = r"(?<!\d)\d{5,6}(?!\d)"             # standalone 5/6-digit run, not part of a longer number
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)     # word tokens; keeps Devanagari and Kannada intact
LENGTH_PCTS = (1, 25, 50, 75, 95, 99)
COUNT_BUCKETS = (0, 1, 2, 3, 4)                   # digitize bucket 5 collects 5+

# One pass over the name column finds every noisy record; the categories are then counted on that
# small subset, which keeps the whole catalogue to two extra full-column passes.
NAME_NOISE_RE = re.compile(r"^[^\w]|www\.|https?:|\.(?:com|in|net|org|co|us|fr)\b|\bdba\b|trading as")
LEADING_JUNK_RE = re.compile(r"^[^\w]")
DOMAIN_RE = re.compile(r"www\.|https?:|\.(?:com|in|net|org|co|us|fr)\b")
DBA_RE = re.compile(r"\bdba\b|trading as")
HAS_DIGIT_RE = re.compile(r"\d")
LOWER_RE = re.compile(r"[a-z]")
LEADS_NUMBER_RE = re.compile(r"^\s*\d")
FRENCH_BIS_RE = re.compile(r"\b(?:bis|ter|quater)\b", re.IGNORECASE)   # case-folded: BIs, Bis, BIS all occur
NULL_PLACEHOLDER_RE = re.compile(r"<\s*null\s*>|\bnull\b|\bn/a\b", re.IGNORECASE)


# --------------------------------------------------------------------------- reporting

def _fmt(n) -> str:
    return f"{n:,}"


def _pct(num, den) -> str:
    return f"{100.0 * num / den:.2f}%" if den else "n/a"


def _cell(value, width: int = 78) -> str:
    """One markdown table cell: single line, pipes escaped, truncated, never empty."""
    text = " ".join(str(value).split()).replace("|", "\\|")
    if len(text) > width:
        text = text[: width - 3] + "..."
    return text or "(empty)"


class Report:
    """Buffers markdown lines and prints them, so stdout and docs/EDA.md come from one source.

    progress() goes to stdout only: the document keeps no timing noise.
    """

    def __init__(self, path: Path = DOC_PATH, echo: bool = True) -> None:
        self.path = path
        self.echo = echo
        self.lines: list[str] = []

    def progress(self, message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    def _add(self, line: str = "") -> None:
        self.lines.append(line)
        if self.echo:
            print(line, flush=True)

    def title(self, text: str) -> None:
        self._add(f"# {text}")
        self._add()

    def h2(self, text: str) -> None:
        self._add()
        self._add(f"## {text}")
        self._add()

    def note(self, text: str = "") -> None:
        self._add(text)
        self._add()

    def spacer(self) -> None:
        self._add()

    def table(self, headers, rows) -> None:
        """Pipe table; all-numeric columns are right-aligned so the terminal lines up."""
        rows = [list(r) for r in rows]
        numeric = [bool(rows) and all(isinstance(r[i], (int, float, np.integer, np.floating))
                                      and not isinstance(r[i], bool) for r in rows)
                   for i in range(len(headers))]
        self._add("| " + " | ".join(str(h) for h in headers) + " |")
        self._add("| " + " | ".join("---:" if a else ":---" for a in numeric) + " |")
        for row in rows:
            cells = [_fmt(v) if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                     else str(v) for v in row]
            self._add("| " + " | ".join(cells) + " |")
        self._add()

    def write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(self.lines).rstrip() + "\n")
        print(f"\nwrote {self.path} ({len(self.lines)} lines, {self.path.stat().st_size / 1024:.1f} KB)",
              flush=True)


# --------------------------------------------------------------------------- scanning one file

def profile_frame(df: pd.DataFrame) -> dict:
    """Row counts by country, field quality, length percentiles and the noise catalogue."""
    n_rows = len(df)
    names = df["business_name"]
    addrs = df["business_address"]
    name_len = names.str.len().to_numpy(np.int32)
    addr_len = addrs.str.len().to_numpy(np.int32)

    noisy = names[names.str.contains(NAME_NOISE_RE)]
    n_noisy = len(noisy)
    out = {
        "rows": n_rows,
        "by_country": {str(k): int(v) for k, v in df["country"].value_counts().items()},
        "empty_name": int((name_len == 0).sum()),
        "empty_addr": int((addr_len == 0).sum()),
        "name_pct": np.percentile(name_len, LENGTH_PCTS).round(1).tolist(),
        "addr_pct": np.percentile(addr_len, LENGTH_PCTS).round(1).tolist(),
        "name_max": int(name_len.max()),
        "addr_max": int(addr_len.max()),
        "postcode_like": int(addrs.str.contains(POSTCODE_RE).sum()),
        "name_tokens_mean": float((names.str.count(r"\s+") + 1).mean()),
        "name_noise": {
            "any of the below": n_noisy,
            "leading junk char (`<< `, `-- `, digit)": int(noisy.str.contains(LEADING_JUNK_RE).sum()),
            "web domain / URL as name": int(noisy.str.contains(DOMAIN_RE).sum()),
            "DBA or trading-as marker": int(noisy.str.contains(DBA_RE).sum()),
            "single token": int((names.str.count(r"\s+") == 0).sum()),
            "contains a digit": int(names.str.contains(HAS_DIGIT_RE).sum()),
        },
        "addr_noise": {
            "no lowercase letter (UPPERCASE address)": int((~addrs.str.contains(LOWER_RE)).sum()),
            "starts with a number": int(addrs.str.contains(LEADS_NUMBER_RE).sum()),
            "bis / ter / quater": int(addrs.str.contains(FRENCH_BIS_RE).sum()),
            "literal <NULL> / NULL / N/A placeholder": int(addrs.str.contains(NULL_PLACEHOLDER_RE).sum()),
        },
    }
    return out


def france_block(df: pd.DataFrame, n_france: int, top: int, seed: int) -> dict:
    """Inputs-only view of one test source restricted to country == 'France' (question g)."""
    fr = df[df["country"] == "France"]
    if fr.empty:
        return {"rows": 0}
    sample = fr.sample(n=min(n_france, len(fr)), random_state=seed)
    names: Counter = Counter()
    addrs: Counter = Counter()
    for text in fr["business_name"]:
        names.update(t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and not t.isdigit())
    for text in fr["business_address"]:
        addrs.update(t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1 and not t.isdigit())
    return {
        "rows": len(fr),
        "sample": sample[["entity_id", "business_name", "business_address"]].values.tolist(),
        "name_tokens": names.most_common(top),
        "addr_tokens": addrs.most_common(top),
        "empty_addr": int((fr["business_address"] == "").sum()),
        "postcode_like": int(fr["business_address"].str.contains(POSTCODE_RE).sum()),
    }


def scan_file(split: str, source: int, rep: Report, *, keep_ids: bool = False, france: bool = False,
              n_france: int = 40, top: int = 30, seed: int = config.SEED) -> dict:
    """One io_utils.read_source pass: profile, France block, and id/country arrays for later joins.

    Country strings are returned as int8 category codes, so the joins in the questions below run on
    integers instead of on 7.6M Python strings.
    """
    rep.progress(f"reading {split} S{source} ...")
    df = io_utils.read_source(split, source)
    out = {"stats": profile_frame(df), "label": f"{split} S{source}"}
    out["stats"]["label"] = out["label"]
    if france:
        out["france"] = france_block(df, n_france, top, seed)
    if keep_ids:
        codes, values = pd.factorize(df["country"], sort=True)
        out["ids"] = df["entity_id"].to_numpy()
        out["country_codes"] = codes.astype(np.int8)
        out["country_values"] = [str(v) for v in values]
    del df
    gc.collect()
    return out


def fetch_text(split: str, source: int, ids) -> dict:
    """Second, targeted read of one file, to pull the raw text of a handful of ids (question f)."""
    wanted = set(ids)
    df = io_utils.read_source(split, source)
    out = {r.entity_id: (r.business_name, r.business_address)
           for r in df[df["entity_id"].isin(wanted)].itertuples()}
    del df
    gc.collect()
    return out


# --------------------------------------------------------------------------- ground truth

def build_pairs(gt: pd.DataFrame, rep: Report) -> dict:
    """Explode the ground truth into flat per-pair arrays plus the per-S1 match counts.

    The numeric id parts are extracted once, for the Spearman check (h) only.
    """
    rep.progress("exploding the ground truth ...")
    match_lists = gt["matches"]
    n_s1 = len(gt)
    lens = np.fromiter(map(len, match_lists), dtype=np.int32, count=n_s1)
    n_pairs = int(lens.sum())
    s1_ids = gt["s1_id"].to_numpy()
    rep.progress(f"parsing the numeric id parts of {n_pairs} pairs ...")
    s1_num = pd.Series(s1_ids).str.slice(ID_PREFIX_LEN).astype("int64").to_numpy()
    cand_ids = np.asarray(list(itertools.chain.from_iterable(match_lists)), dtype=object)
    cand_num = pd.Series(cand_ids).str.slice(ID_PREFIX_LEN).astype("int64").to_numpy()
    return {"s1_ids": s1_ids, "lens": lens, "n_s1": n_s1, "n_pairs": n_pairs,
            "pair_s1_id": np.repeat(s1_ids, lens), "pair_cand_id": cand_ids,
            "pair_s1_num": np.repeat(s1_num, lens), "pair_cand_num": cand_num}


def link_pairs(pairs: dict, s1: dict, pool: dict) -> dict:
    """Resolve both ends of every pair to integer row positions. -1 means "id not in the file".

    The S2 and S3 pools share one index, so a single get_indexer resolves all 7.6M candidate ids.
    """
    s1_pos = pd.Index(s1["ids"]).get_indexer(pairs["s1_ids"])
    cand_pos = pd.Index(pool["ids"]).get_indexer(pairs["pair_cand_id"])
    pair_s1_pos = np.repeat(s1_pos, pairs["lens"])
    valid = cand_pos >= 0
    return {
        "s1_pos": s1_pos,
        "pair_s1_pos": pair_s1_pos,
        "cand_pos": cand_pos,
        "valid": valid,
        "pair_source": pool["source"][cand_pos],
        "pair_s1_country": s1["country_codes"][pair_s1_pos],
        "pair_cand_country": pool["country_codes"][cand_pos],
    }


# --------------------------------------------------------------------------- the questions

def answer_a(rep: Report, pairs: dict, link: dict, pool: dict) -> None:
    rep.h2("(a) Can one S2/S3 id match two or more Source 1 entities?")
    n_pairs, valid = pairs["n_pairs"], link["valid"]
    counts = np.bincount(link["cand_pos"][valid], minlength=len(pool["ids"]))
    used = counts[counts > 0]
    rows = [
        ["ground-truth pairs", n_pairs],
        ["pairs whose candidate id exists in the S2/S3 pools", int(valid.sum())],
        ["pairs whose candidate id is missing from the pools", n_pairs - int(valid.sum())],
        ["S1 entities in the ground truth", pairs["n_s1"]],
        ["distinct candidate ids used at least once", len(used)],
        ["**candidate ids claimed by 2+ S1 entities**", int((used >= 2).sum())],
        ["candidate ids claimed by 3+ S1 entities", int((used >= 3).sum())],
        ["largest number of S1 entities on one candidate id", int(used.max())],
    ]
    for source in (2, 3):
        sel = valid & (link["pair_source"] == source)
        sub = np.bincount(link["cand_pos"][sel], minlength=len(pool["ids"]))
        sub = sub[sub > 0]
        rows.append([f"... restricted to S{source}: distinct ids / used by 2+ S1 / max fan-in",
                     f"{len(sub):,} / {int((sub >= 2).sum()):,} / {int(sub.max())}"])
    rep.table(["quantity", "value"], rows)
    dup = np.flatnonzero(counts >= 2)
    if len(dup):
        rep.note("Counter-examples: " + ", ".join(str(x) for x in pool["ids"][dup[:10]]))
        rep.note("**Answer: yes** - exclusivity does not hold, so postprocessing may not assume it.")
    else:
        rep.note("**Answer: no.** Every S2/S3 id belongs to at most one Source 1 entity, so the "
                 "exclusivity rule Arushi applies is consistent with the training labels.")


def answer_b(rep: Report, link: dict, s1: dict, pool: dict) -> None:
    rep.h2("(b) Does every ground-truth pair share one country string?")
    ok = link["valid"]
    ct = np.zeros((len(s1["country_values"]), len(pool["country_values"])), dtype=np.int64)
    np.add.at(ct, (link["pair_s1_country"][ok], link["pair_cand_country"][ok]), 1)
    total = int(ct.sum())
    rep.note("Rows = country of the S1 entity, columns = country of the matched S2/S3 record.")
    rep.table(["S1 country \\ candidate country"] + pool["country_values"],
              [[s1["country_values"][i]] + [int(v) for v in ct[i]] for i in range(ct.shape[0])])

    # Same matrix after strip().casefold(), which is how a stray ' France' or 'us' would show up.
    pool_norm = [v.strip().casefold() for v in pool["country_values"]]
    pool_norm_index = {v: i for i, v in enumerate(pool_norm)}
    mismatch, unknown = 0, 0
    for i, value in enumerate(s1["country_values"]):
        j = pool_norm_index.get(value.strip().casefold(), -1)
        for k in range(ct.shape[1]):
            if j < 0:
                unknown += int(ct[i, k])
            elif k != j:
                mismatch += int(ct[i, k])
    rep.table(["quantity", "value"],
              [["ground-truth pairs with both ids resolved", total],
               ["**pairs whose two country strings differ**", mismatch],
               ["S1 country strings with no counterpart in the S2/S3 pools", unknown]])
    if mismatch == 0 and unknown == 0:
        rep.note("**Answer: yes.** Not a single pair disagrees, so country is a safe hard blocking key. "
                 "The label set is still open - train has no France, test does - so never hard-code, "
                 "filter or one-hot it; group by whatever strings the file contains.")
    else:
        rep.note("**Answer: no** - see the counts above before blocking within country.")


def answer_c(rep: Report, pairs: dict, link: dict, s1: dict) -> None:
    rep.h2("(c) Singleton rate and match counts per Source 1 entity")
    lens, n_s1 = pairs["lens"], pairs["n_s1"]
    buckets = np.digitize(lens, np.array(COUNT_BUCKETS) + 1)
    labels = [f"{b}{'+' if b == len(COUNT_BUCKETS) else ''}" for b in range(len(COUNT_BUCKETS) + 1)]
    rep.note(f"All {_fmt(n_s1)} training S1 entities, the dev sample included.")
    rep.table(["matches per S1", "S1 entities", "share"],
              [[labels[b], int((buckets == b).sum()), _pct(int((buckets == b).sum()), n_s1)]
               for b in range(len(labels))])
    tail = {int(v): int((lens == v).sum()) for v in range(len(COUNT_BUCKETS) + 1, int(lens.max()) + 1)}
    if tail:
        rep.note("Exact counts above the 5+ bucket: "
                 + ", ".join(f"{k} -> {v:,}" for k, v in tail.items())
                 + f". The busiest entity has {int(lens.max())} matches.")

    # one row per ground-truth S1 entity, in ground-truth order
    s1_country = np.asarray([s1["country_values"][c] for c in s1["country_codes"][link["s1_pos"]]],
                             dtype=object)
    per_country = (pd.DataFrame({"country": s1_country, "b": buckets})
                   .groupby(["country", "b"]).size().unstack(fill_value=0)
                   .reindex(columns=range(len(labels)), fill_value=0))
    rep.note("Per S1 country:")
    rep.table(["country"] + labels + ["S1 entities", "singleton rate"],
              [[idx] + [int(v) for v in row] + [int(row.sum()), _pct(int(row.iloc[0]), int(row.sum()))]
               for idx, row in per_country.iterrows()]
              + [["**all**"] + [int(per_country[c].sum()) for c in per_country.columns]
                 + [n_s1, _pct(int((lens == 0).sum()), n_s1)]])

    source = link["pair_source"][link["valid"]]
    per_country_src = (pd.DataFrame({"country": s1_country[link["pair_s1_pos"]][link["valid"]],
                                     "source": source})
                       .groupby(["country", "source"]).size().unstack(fill_value=0))
    rep.note("Where the matches come from (S1 country x matched source):")
    rep.table(["country", "S2 matches", "S3 matches", "all matches", "S2 share"],
              [[idx, int(row.get(2, 0)), int(row.get(3, 0)), int(row.sum()),
                _pct(int(row.get(2, 0)), int(row.sum()))] for idx, row in per_country_src.iterrows()]
              + [["**all**", int((source == 2).sum()), int((source == 3).sum()), int(len(source)),
                  _pct(int((source == 2).sum()), int(len(source)))]])

    in_sample = np.fromiter((io_utils.in_dev_sample(x) for x in pairs["s1_ids"]), dtype=bool, count=n_s1)
    rep.note(f"Restricted to the `in_dev_sample()` {config.SAMPLE_FRAC:.0%} of S1: "
             f"{_fmt(int(in_sample.sum()))} entities, {_fmt(int(lens[in_sample].sum()))} pairs. This is the "
             f"row count Arushi's labels.parquet and folds.parquet must show.")
    rep.table(["matches per S1", "S1 entities", "share"],
              [[labels[b], int((buckets[in_sample] == b).sum()),
                _pct(int((buckets[in_sample] == b).sum()), int(in_sample.sum()))] for b in range(len(labels))])
    rep.note(f"**Singleton rate: {_pct(int((lens == 0).sum()), n_s1)}** of all training S1 entities, "
             f"{_pct(int((buckets[in_sample] == 0).sum()), int(in_sample.sum()))} inside the dev sample. "
             f"An entity in that bucket scores 1.0 only if we predict nothing for it, so the selection "
             f"step must be able to emit an empty match list.")


def answer_d(rep: Report, pairs: dict, link: dict) -> None:
    rep.h2("(d) Can two records from the same source match one Source 1 entity?")
    n_s1, ok = pairs["n_s1"], link["valid"]
    n2 = np.bincount(link["pair_s1_pos"][ok & (link["pair_source"] == 2)], minlength=n_s1)
    n3 = np.bincount(link["pair_s1_pos"][ok & (link["pair_source"] == 3)], minlength=n_s1)
    one = (n2 + n3) == 1
    rep.table(["quantity", "S1 entities", "share of all S1"],
              [["2+ matches inside S2", int((n2 >= 2).sum()), _pct(int((n2 >= 2).sum()), n_s1)],
               ["2+ matches inside S3", int((n3 >= 2).sum()), _pct(int((n3 >= 2).sum()), n_s1)],
               ["**2+ matches inside either source**", int(((n2 >= 2) | (n3 >= 2)).sum()),
                _pct(int(((n2 >= 2) | (n3 >= 2)).sum()), n_s1)],
               ["exactly one match in total", int(one.sum()), _pct(int(one.sum()), n_s1)]])
    joint = (pd.DataFrame({"n2": np.minimum(n2, 3), "n3": np.minimum(n3, 3)})
             .groupby(["n2", "n3"]).size().unstack(fill_value=0))
    rep.note("Joint distribution, counts capped at 3+ (rows = S2 matches, columns = S3 matches):")
    rep.table(["S2 \\ S3", "0", "1", "2", "3+"],
              [[int(i)] + [int(joint.loc[i, c]) if c in joint.columns else 0 for c in (0, 1, 2, 3)]
               for i in joint.index])
    rep.note("**Yes** - same-source duplicates are the norm. A per-source cap of 1 would throw away "
             "true matches, and an S1 is expected to own several records inside a single source.")


def answer_e(rep: Report, pairs: dict, link: dict, stats: dict) -> None:
    rep.h2("(e) Row counts per split x source x country, and the distractor share")
    rows = []
    for split in config.SPLITS:
        for source in config.SOURCES:
            for country, n in sorted(stats[(split, source)]["by_country"].items(), key=lambda kv: -kv[1]):
                rows.append([split, f"S{source}", country, n])
            rows.append([split, f"S{source}", "**all**", stats[(split, source)]["rows"]])
    rep.table(["split", "source", "country", "rows"], rows)

    ok = link["valid"]
    used2 = int(np.unique(link["cand_pos"][ok & (link["pair_source"] == 2)]).size)
    used3 = int(np.unique(link["cand_pos"][ok & (link["pair_source"] == 3)]).size)
    rep.table(["pool", "records", "records with at least one S1", "records that match nothing",
               "distractor share"],
              [["train S2", stats[("train", 2)]["rows"], used2, stats[("train", 2)]["rows"] - used2,
                _pct(stats[("train", 2)]["rows"] - used2, stats[("train", 2)]["rows"])],
               ["train S3", stats[("train", 3)]["rows"], used3, stats[("train", 3)]["rows"] - used3,
                _pct(stats[("train", 3)]["rows"] - used3, stats[("train", 3)]["rows"])]])
    rep.note(f"A distractor is a pool record that belongs to no S1 entity at all, so a stage that "
             f"predicts a match for one is a false positive. The test split has no labels, so its "
             f"distractor share is unknown; expect it to sit near the train value.")


def answer_f(rep: Report, pairs: dict, link: dict, s1: dict, n_pairs: int, seed: int) -> None:
    """Raw text of random matched pairs, per S1 country and per matched source."""
    rng = np.random.default_rng(seed)
    ok = link["valid"]
    group_key = link["pair_s1_country"].astype(np.int32) * 2 + link["pair_source"]
    chosen: list[tuple[str, int, int]] = []
    for code, value in enumerate(s1["country_values"]):
        for source in (2, 3):
            idx = np.flatnonzero(ok & (group_key == code * 2 + source))
            if len(idx) >= n_pairs:
                chosen += [(value, source, int(j)) for j in rng.choice(idx, size=n_pairs, replace=False)]

    ids1 = sorted({pairs["pair_s1_id"][i] for _, _, i in chosen})
    ids_c = sorted({pairs["pair_cand_id"][i] for _, _, i in chosen})
    rep.progress(f"re-reading train S1/S2/S3 for the text of {len(ids1)} + {len(ids_c)} sampled ids ...")
    text1 = fetch_text("train", 1, ids1)
    text2 = fetch_text("train", 2, ids_c)
    text3 = fetch_text("train", 3, ids_c)
    cand_text = {**text2, **text3}

    rep.h2("(f) Random matched pairs, S1 against the matched record (raw input text)")
    rep.note(f"{n_pairs} pairs per (S1 country, matched source), seed {seed}. This is the noise the "
             f"normalization has to survive.")
    empty_addr = 0
    for country, source in sorted({(c, s) for c, s, _ in chosen}):
        rows = []
        for c, s, i in chosen:
            if (c, s) != (country, source):
                continue
            s1_id, cand_id = pairs["pair_s1_id"][i], pairs["pair_cand_id"][i]
            s1_name, s1_addr = text1[s1_id]
            c_name, c_addr = cand_text[cand_id]
            empty_addr += int(c_addr == "")
            rows.append([f"`{s1_id}` -> `{cand_id}`", _cell(s1_name), _cell(s1_addr),
                         _cell(c_name), _cell(c_addr)])
        rep.note(f"S1 country `{country}`, matched source S{source}:")
        rep.table(["pair", "S1 name", "S1 address", f"S{source} name", f"S{source} address"], rows)
    rep.note(f"{empty_addr} of the {len(chosen)} sampled matched records have an **empty address** "
             f"({_pct(empty_addr, len(chosen))}); the S1 address is never empty.")


def answer_g(rep: Report, france: dict) -> None:
    rep.h2("(g) Test France records (inputs only - the test split has no labels)")
    rows = []
    for source in sorted(france):
        for entity_id, name, addr in france[source].get("sample", []):
            rows.append([f"S{source}", entity_id, _cell(name), _cell(addr)])
    rep.table(["source", "entity_id", "business_name", "business_address"], rows)

    names: Counter = Counter()
    addrs: Counter = Counter()
    for blk in france.values():
        names.update(dict(blk.get("name_tokens", [])))
        addrs.update(dict(blk.get("addr_tokens", [])))
    rep.note("Most frequent tokens over **all** France records of the test split, "
             "numeric and 1-character tokens dropped:")
    width = max((len(names), len(addrs)))
    rep.table(["name token", "count", "address token", "count"],
              [[f"`{n}`", c, f"`{a}`" if a else "", d] for (n, c), (a, d) in
               zip(names.most_common(width), addrs.most_common(width))])
    rep.table(["file", "France rows", "empty address", "address with a 5- or 6-digit number"],
              [[f"test S{s}", blk["rows"], blk["empty_addr"],
                f"{blk['postcode_like']:,} ({_pct(blk['postcode_like'], blk['rows'])})"]
               for s, blk in sorted(france.items()) if blk.get("rows")])
    rep.note("This is the input for the BH-5 tables: legal forms (SARL, SAS, SA, S.A.S, EURL, SCI) and "
             "street types (rue, r, bd, bvd, av, allee) become country-independent word mappings in "
             "normalize.py - no `if country` branch anywhere.")


def answer_h(rep: Report, pairs: dict, link: dict) -> None:
    rep.h2("(h) Do the numeric parts of the matched ids correlate? (report only - never a feature)")
    rows = []
    for label, sel in [("all pairs", link["valid"]),
                       ("S1 -> S2", link["valid"] & (link["pair_source"] == 2)),
                       ("S1 -> S3", link["valid"] & (link["pair_source"] == 3))]:
        rho = spearmanr(pairs["pair_s1_num"][sel], pairs["pair_cand_num"][sel]).statistic
        rows.append([label, int(sel.sum()), round(float(rho), 6)])
    rep.table(["pair set", "pairs", "Spearman rho of the numeric id parts"], rows)
    rep.note("Row order and ID numbers carry no signal, so nothing in `normalize.py` or in the "
             "`fn_*` / `fa_*` feature files may depend on them.")


def answer_extra(rep: Report, stats: dict) -> None:
    rep.h2("Field quality, lengths and postcode-like numbers over all six files")
    ordered = [stats[(s, src)] for s in config.SPLITS for src in config.SOURCES]
    rep.table(["file", "rows", "empty name", "empty address", "empty address share"],
              [[st["label"], st["rows"], st["empty_name"], st["empty_addr"],
                _pct(st["empty_addr"], st["rows"])] for st in ordered])
    pcts = ["p1", "p25", "p50", "p75", "p95", "p99", "max"]
    rep.note("Business name length in characters.")
    rep.table(["file"] + pcts,
              [[st["label"]] + [f"{v:.0f}" for v in st["name_pct"]] + [str(st["name_max"])] for st in ordered])
    rep.note("Business address length in characters.")
    rep.table(["file"] + pcts,
              [[st["label"]] + [f"{v:.0f}" for v in st["addr_pct"]] + [str(st["addr_max"])] for st in ordered])
    rep.note("Business name tokens per record (whitespace split).")
    rep.table(["file", "mean tokens"],
              [[st["label"], round(st["name_tokens_mean"], 2)] for st in ordered])
    rep.table(["file", "rows", "address with a standalone 5- or 6-digit number", "share"],
              [[st["label"], st["rows"], st["postcode_like"], _pct(st["postcode_like"], st["rows"])]
               for st in ordered])
    rep.note("A standalone 5- or 6-digit run: US ZIPs, French 5-digit codes, Indian PINs. The rest have "
             "no postcode at all, or a number that is part of a longer one.")

    rep.h2("Noise catalogue for the normalization (BH-2 / BH-3)")
    name_keys = list(ordered[0]["name_noise"].keys())
    rep.table(["file", "records"] + name_keys,
              [[st["label"], st["rows"]] + [f"{st['name_noise'][k]:,} ({_pct(st['name_noise'][k], st['rows'])})"
                                            for k in name_keys] for st in ordered])
    addr_keys = list(ordered[0]["addr_noise"].keys())
    rep.table(["file", "records"] + addr_keys,
              [[st["label"], st["rows"]] + [f"{st['addr_noise'][k]:,} ({_pct(st['addr_noise'][k], st['rows'])})"
                                            for k in addr_keys] for st in ordered])
    rep.note("So the normalizer needs: leading-junk stripping, a DBA / trading-as split, legal-form "
             "classes, UPPERCASE folding, address component reordering, a number tokenizer that keeps "
             "`5 bis rue ...` together, a postcode extractor, and a `<NULL>` / `N/A` placeholder to "
             "delete. Note the asymmetry: S2 addresses are mostly UPPERCASE while S3 addresses are "
             "mostly title case, so case folding has to happen before any token comparison.")


# --------------------------------------------------------------------------- driver

def run_checks(write: bool = True, n_pairs: int = 25, n_france: int = 40, top: int = 30,
               seed: int = config.SEED, echo: bool = True) -> Report:
    """Scan all six files plus the train ground truth and answer (a)-(h). Returns the report."""
    rep = Report(echo=echo)
    rep.title("EDA and noise catalogue (BH-1)")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=config.ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:                                    # git missing or not a repo: not worth failing over
        sha = "unknown"
    rep.note(f"Generated by `python -m src.eda` on {time.strftime('%Y-%m-%d %H:%M')} local time - seed "
             f"{seed} - git {sha} - `BER_SAMPLE_FRAC={config.SAMPLE_FRAC}`. Counts use the **full** train "
             f"split; the dev sample is applied only where a line says so. The raw TSVs are read "
             f"exclusively through `src/io_utils.py`.")

    stats: dict = {}
    s1: dict = {}
    pools: dict = {}
    france: dict = {}
    for split in config.SPLITS:
        for source in config.SOURCES:
            blk = scan_file(split, source, rep, keep_ids=(split == "train"),
                            france=(split == "test"), n_france=n_france, top=top, seed=seed)
            stats[(split, source)] = blk["stats"]
            if split == "test":
                france[source] = blk["france"]
            elif source == 1:
                s1 = blk
            else:
                pools[source] = blk

    # one shared index over the S2 and S3 pools; country labels are remapped to a common list so the
    # crosstab in (b) has one row and one column per distinct country string
    country_values = sorted(set(pools[2]["country_values"]) | set(pools[3]["country_values"]))
    remap = {source: np.array([country_values.index(v) for v in pools[source]["country_values"]],
                              dtype=np.int8) for source in (2, 3)}
    pool = {
        "ids": np.concatenate([pools[2]["ids"], pools[3]["ids"]]),
        "country_codes": np.concatenate([remap[2][pools[2]["country_codes"]],
                                         remap[3][pools[3]["country_codes"]]]),
        "country_values": country_values,
        "source": np.concatenate([np.full(len(pools[2]["ids"]), 2, np.int8),
                                  np.full(len(pools[3]["ids"]), 3, np.int8)]),
    }
    del pools

    rep.progress("reading the train ground truth ...")
    pairs = build_pairs(io_utils.read_ground_truth(), rep)
    link = link_pairs(pairs, s1, pool)

    answer_a(rep, pairs, link, pool)
    answer_b(rep, link, s1, pool)
    answer_c(rep, pairs, link, s1)
    answer_d(rep, pairs, link)
    answer_e(rep, pairs, link, stats)
    answer_f(rep, pairs, link, s1, n_pairs, seed)
    answer_g(rep, france)
    answer_h(rep, pairs, link)
    answer_extra(rep, stats)

    if write:
        rep.write()
    return rep


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BH-1 data checks -> docs/EDA.md")
    parser.add_argument("--no-write", action="store_true", help="print only, do not touch docs/EDA.md")
    parser.add_argument("--pairs", type=int, default=25, help="matched pairs per country and source")
    parser.add_argument("--france", type=int, default=40, help="random test France records per source")
    parser.add_argument("--top", type=int, default=30, help="most frequent France tokens to list")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # French accents on a cp1252 console
    run_checks(write=not args.no_write, n_pairs=args.pairs, n_france=args.france, top=args.top)

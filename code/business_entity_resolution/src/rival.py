"""
Same-name / same-address RIVAL features (the ens3 stage-2 upgrade, ported from the
kaggle/kernels/ens3 kernel; found with the ctx2/ctx3 re-scorer: +0.0015 honest on the holdout).

40% of Source 1 names occur more than once (one business name, several addresses), and many
Source 1 addresses hold several businesses. A Source 2/3 copy with an empty address (or with a
made-up name) fits every such rival about equally, so a pair model that looks at ONE pair gives
each ~1/n. The copy was made from ONE of them: its exact spelling ("Ltd" vs "Limited", "Inc" or
not, the numbers of the address) points at the owner. For every unsure pair these features compare
the copy with ALL Source 1 records of the split that share its name key (or its address numbers).
They are counted over the FULL Source 1 table of the split (train or test), so the 25% training
sample does not change them: train and test get the same kind of numbers.
Only pairs with a stage-1 probability in (LOW, HIGH) get them (the others: NaN), in train and test alike.

src version (differences from the Kaggle blob):
  - the tables come from the cleaned parquet files of the split (load_tables), not by re-reading
    the raw .tsv files, so the cleaned text (transliteration, legal forms) is shared with the rest
    of the pipeline; `data_dir` is only needed when the cleaned tables do not exist yet
  - exact behaviour of the features is unchanged: RIVAL_COLUMNS, the unsure window and the
    per-pair rival comparison are the same as the kernel that produced holdout 0.9874
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from io_utils import log

LOW, HIGH = 0.005, 0.995
MAX_RIVALS = 300
LEGAL = set("llc inc ltd limited pvt private corp corporation co company the lp llp pc pa plc pllc sarl sas "
            "sasu eurl sa sci and of".split())
RIVAL_COLUMNS = ["rv_empty", "rv_key_eq", "rv_n1", "rv_n2", "rv_sim", "rv_tset", "rv_gap_best", "rv_n_better",
                 "rv_n_tie", "rv_gap_second", "rv_asim", "rv_ad_n2", "rv_ad_gap_best", "rv_ad_better",
                 "rv_ad_gap_second", "rv_n_words", "rv_max_word"]
_TABLES = {}


def plain(texts):
    s = pd.Series(texts, dtype=object).fillna("").str.lower()
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def name_key(pl):
    s = pl.str.replace(r"www\.\S+|https?://\S+|\S+@\S+|\S+\.com\b", " ", regex=True)
    s = s.str.replace(r"\(id:?\s*\d+\)|\d{6,}", " ", regex=True)
    s = s.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    s = s.str.replace(r"\b(\w+)( \1\b)+", r"\1", regex=True)
    return s.map(lambda x: " ".join(w for w in x.split() if w not in LEGAL))


def digit_key(addr):
    return addr.str.findall(r"\d+").map(lambda x: " ".join(sorted(set(x))))


def _groups(keys):
    """key -> (sorted row numbers, bounds) without a Python dict of arrays."""
    codes, uniques = pd.factorize(keys)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return pd.Index(uniques), order, bounds


def tables_from(s1, s23):
    """Rival tables from the cleaned tables of one split (normalize.py output)."""
    s1_plain = plain(s1["business_name"].to_numpy())
    s1_addr = plain(s1["business_address"].to_numpy())
    t = {"s1_plain": s1_plain, "s1_addr": s1_addr,
         "s23_name": pd.Series(np.asarray(s23["business_name"], dtype=object)),
         "s23_addr": pd.Series(np.asarray(s23["business_address"], dtype=object))}
    t["s1_key"] = name_key(s1_plain)
    t["s1_dkey"] = digit_key(s1_addr)
    t["name_groups"] = _groups(t["s1_key"].to_numpy())
    t["addr_groups"] = _groups(t["s1_dkey"].to_numpy())
    t["n_key"] = t["s1_key"].map(t["s1_key"].value_counts()).to_numpy(dtype=np.float32)
    return t


def load_tables(split_dir):
    """Full Source 1 and Source 2/3 of one split, from its cleaned tables (cached).
    split_dir: the folder with s1_clean.parquet / s23_clean.parquet (<work>/train or <work>/test)."""
    split_dir = Path(split_dir)
    if _TABLES.get("split") == split_dir:
        return _TABLES["t"]
    s1 = pd.read_parquet(split_dir / "s1_clean.parquet", columns=["business_name", "business_address"])
    s23 = pd.read_parquet(split_dir / "s23_clean.parquet", columns=["business_name", "business_address"])
    t = tables_from(s1, s23)
    log(f"rival: tables loaded ({len(s1):,} Source 1, {len(s23):,} Source 2/3 records)")
    _TABLES.clear()
    _TABLES.update({"split": split_dir, "t": t})
    return t


def release():
    _TABLES.clear()


def _rivals(query, own, groups, key, s1_texts, scorer, own_score):
    """Scores of the query text against every Source 1 record with this key."""
    index, order, bounds = groups
    code = index.get_indexer([key])[0]
    if code < 0:
        return None
    rs = order[bounds[code]:bounds[code + 1]]
    if len(rs) == 0 or len(rs) > MAX_RIVALS:
        return None
    sc = np.fromiter((scorer(query, s1_texts[r]) for r in rs), dtype=np.float32, count=len(rs))
    others = sc[rs != own]
    return sc.max(), (sc > own_score).sum(), (sc == own_score).sum() - (own in rs), (others.max() if len(others) else np.nan)


def rival_features(s1_row, s23_row, p, tables):
    """dict RIVAL_COLUMNS -> float32 arrays (NaN outside the unsure window).
    `tables` is the output of tables_from / load_tables for this split."""
    n = len(p)
    out = {c: np.full(n, np.nan, dtype=np.float32) for c in RIVAL_COLUMNS}
    w = np.flatnonzero((p > LOW) & (p < HIGH))
    if len(w) == 0:
        return out
    a = s1_row[w]
    u, inv = np.unique(s23_row[w], return_inverse=True)
    r_plain = plain(tables["s23_name"].take(u).to_numpy())
    r_addr = plain(tables["s23_addr"].take(u).to_numpy())
    r_key, r_dkey = name_key(r_plain), digit_key(r_addr)
    b_plain, b_key = r_plain.to_numpy()[inv], r_key.to_numpy()[inv]
    b_addr, b_dkey = r_addr.to_numpy()[inv], r_dkey.to_numpy()[inv]
    s1_plain, s1_key, s1_addr = tables["s1_plain"].to_numpy(), tables["s1_key"].to_numpy(), tables["s1_addr"].to_numpy()
    s1_dkey = tables["s1_dkey"].to_numpy()
    a_plain, a_key, a_addr = s1_plain[a], s1_key[a], s1_addr[a]
    empty = (b_addr == "").astype(np.float32)
    sim = np.array([fuzz.ratio(x, y) for x, y in zip(a_plain, b_plain)], dtype=np.float32)
    asim = np.array([fuzz.token_set_ratio(x, y) for x, y in zip(a_addr, b_addr)], dtype=np.float32)
    kc = tables["s1_key"].value_counts()
    n2 = pd.Series(b_key).map(kc).fillna(0).to_numpy(dtype=np.float32)
    dc = tables["s1_dkey"].value_counts()
    ad_n2 = pd.Series(b_dkey).map(dc).fillna(0).to_numpy(dtype=np.float32)
    ad_n2[b_dkey == ""] = -1
    cols = {c: np.full(len(w), np.nan, dtype=np.float32) for c in RIVAL_COLUMNS}
    for i in range(len(w)):
        r = _rivals(b_plain[i], a[i], tables["name_groups"], b_key[i], s1_plain, fuzz.ratio, sim[i])
        if r is not None:
            cols["rv_gap_best"][i], cols["rv_n_better"][i], cols["rv_n_tie"][i] = sim[i] - r[0], r[1], r[2]
            cols["rv_gap_second"][i] = sim[i] - r[3]
        if b_dkey[i]:
            r = _rivals(b_addr[i], a[i], tables["addr_groups"], b_dkey[i], s1_addr, fuzz.token_set_ratio, asim[i])
            if r is not None:
                cols["rv_ad_gap_best"][i], cols["rv_ad_better"][i] = asim[i] - r[0], r[1]
                cols["rv_ad_gap_second"][i] = asim[i] - r[3]
    words = pd.Series(b_plain).str.split()
    cols.update({
        "rv_empty": empty, "rv_key_eq": (a_key == b_key).astype(np.float32), "rv_n1": tables["n_key"][a],
        "rv_n2": n2, "rv_sim": sim,
        "rv_tset": np.array([fuzz.token_set_ratio(x, y) for x, y in zip(a_key, b_key)], dtype=np.float32),
        "rv_asim": asim, "rv_ad_n2": ad_n2,
        "rv_n_words": words.str.len().to_numpy(dtype=np.float32),
        "rv_max_word": words.map(lambda x: max((len(y) for y in x), default=0)).to_numpy(dtype=np.float32)})
    for c in RIVAL_COLUMNS:
        out[c][w] = cols[c]
    log(f"rival: features for {len(w):,} of {n:,} pairs")
    return out

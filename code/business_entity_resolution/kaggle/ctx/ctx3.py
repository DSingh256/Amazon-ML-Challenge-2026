"""
ctx3 = ctx2 + SAME-ADDRESS RIVALS (numbers of the address) + made-up-name signals.
ctx2: SAME-NAME RIVALS re-scorer (local, CPU).
40% of Source 1 names occur more than once (one business name, several addresses). A Source 2/3 copy
with an empty address fits every one of them equally, and the pair model (which judges one pair at a
time) gives each ~1/n. The copy was made from ONE of them, so its exact spelling ("Ltd" vs "Limited",
"Inc" or not) points at the owner. ctx2 adds features that compare the copy with ALL Source 1 records
sharing its name key, counted over the FULL Source 1 of the split (so train and test agree; the 25%
training sample does not matter), plus ctx1's candidate-list features, and learns a better keep score.
Trained on dec1's holdout pairs (truth known), 2-fold by record for an honest check.
Usage: python ctx/ctx2.py [--test <test_pair_probabilities.parquet> <out_dir>]
"""
import json
import re
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from ctx1 import exclusive_sorted, f05, keep_rule, logit  # noqa: E402

KB = Path("D:/AmazonML/kaggle_build")
DEC = KB / "out_dec1/work/dec"
DATA = Path("D:/AmazonML/student_resource/dataset")
OUT = KB / "out_ctx3"
LOW, HIGH = 0.005, 0.995          # pairs below LOW are dropped (both sides), above HIGH kept as they are
T0 = time.time()
LOG = []

LEGAL = set("llc inc ltd limited pvt private corp corporation co company the lp llp pc pa plc pllc sarl sas "
            "sasu eurl sa sci and of".split())


def say(m):
    m = f"[{time.time() - T0:6.0f}s] {m}"
    print(m, flush=True)
    LOG.append(m)
    OUT.mkdir(exist_ok=True)
    (OUT / "ctx3_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def read_cols(path, cols):
    t = pc.read_csv(path, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                    convert_options=pc.ConvertOptions(include_columns=cols,
                                                      column_types={c: pa.string() for c in cols}))
    return {c: t.column(c).combine_chunks() for c in cols}


def plain(names):
    s = pd.Series(names, dtype=object).fillna("").str.lower()
    s = s.str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def name_key(pl):
    s = pl.str.replace(r"www\.\S+|https?://\S+|\S+@\S+|\S+\.com\b", " ", regex=True)
    s = s.str.replace(r"\(id:?\s*\d+\)|\d{6,}", " ", regex=True)
    s = s.str.replace(r"[^a-z0-9 ]", " ", regex=True)
    s = s.str.replace(r"\b(\w+)( \1\b)+", r"\1", regex=True)
    return s.map(lambda x: " ".join(w for w in x.split() if w not in LEGAL))


def digit_key(addr):
    """Sorted numbers of an address ("" when it has none)."""
    return addr.str.findall(r"\d+").map(lambda x: " ".join(sorted(set(x))))


def split_data(split):
    """Plain names / keys / empty-address flags of Source 1 (all) and Source 2+3 (all)."""
    d = DATA / split
    s1 = read_cols(d / f"{split}_source1.tsv", ["business_name", "business_address"])
    s1_plain = plain(s1["business_name"].to_numpy(zero_copy_only=False))
    s1_addr = plain(s1["business_address"].to_numpy(zero_copy_only=False))
    del s1
    s1_key = name_key(s1_plain)
    names, empty, addrs = [], [], []
    for k in (2, 3):
        c = read_cols(d / f"{split}_source{k}.tsv", ["business_name", "business_address"])
        names.append(c["business_name"])
        addrs.append(c["business_address"])
        a = pa.compute.utf8_trim_whitespace(c["business_address"])
        empty.append(np.asarray(pa.compute.equal(a, "").to_numpy(zero_copy_only=False), dtype=bool))
        del c, a
    return s1_plain, s1_key, pa.concat_arrays(names), np.concatenate(empty), s1_addr, pa.concat_arrays(addrs)


def text_features(s1_row, s23_row, s1_plain, s1_key, s23_names, s23_empty, s1_addr, s23_addrs):
    say(f"text features for {len(s1_row):,} pairs")
    n1 = s1_key.map(s1_key.value_counts())
    groups = pd.Series(np.arange(len(s1_key))).groupby(s1_key.to_numpy()).apply(lambda x: x.to_numpy())
    u, inv = np.unique(s23_row, return_inverse=True)
    r_plain = plain(s23_names.take(pa.array(u)).to_numpy(zero_copy_only=False))
    r_key = name_key(r_plain)
    a_plain, a_key = s1_plain.to_numpy()[s1_row], s1_key.to_numpy()[s1_row]
    b_plain, b_key = r_plain.to_numpy()[inv], r_key.to_numpy()[inv]
    sim = np.array([fuzz.ratio(a, b) for a, b in zip(a_plain, b_plain)], dtype=np.float32)
    sim_key = np.array([fuzz.ratio(a, b) for a, b in zip(a_key, b_key)], dtype=np.float32)
    tset = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(a_key, b_key)], dtype=np.float32)
    # rivals: every Source 1 record whose name key equals the copy's key
    kc = s1_key.value_counts()
    n2 = pd.Series(b_key).map(kc).fillna(0).to_numpy(dtype=np.float32)
    best = np.full(len(s1_row), np.nan, dtype=np.float32)
    n_better = np.full(len(s1_row), np.nan, dtype=np.float32)
    n_tie = np.full(len(s1_row), np.nan, dtype=np.float32)
    second = np.full(len(s1_row), np.nan, dtype=np.float32)
    plain_arr = s1_plain.to_numpy()
    for i in np.flatnonzero((n2 >= 1) & (n2 <= 300)):
        rs = groups[b_key[i]]
        sc = np.fromiter((fuzz.ratio(b_plain[i], plain_arr[r]) for r in rs), dtype=np.float32, count=len(rs))
        best[i] = sc.max()
        n_better[i] = (sc > sim[i]).sum()
        n_tie[i] = (sc == sim[i]).sum() - (s1_key.iat[s1_row[i]] == b_key[i])
        others = np.sort(sc[rs != s1_row[i]])
        second[i] = others[-1] if len(others) else np.nan
    say("address rival features")
    s1_dkey = digit_key(s1_addr)
    r_addr = plain(s23_addrs.take(pa.array(u)).to_numpy(zero_copy_only=False))
    b_addr = r_addr.to_numpy()[inv]
    b_dkey = digit_key(r_addr).to_numpy()[inv]
    a_addr = s1_addr.to_numpy()[s1_row]
    asim = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(a_addr, b_addr)], dtype=np.float32)
    dc = s1_dkey.value_counts()
    ad_n2 = pd.Series(b_dkey).map(dc).fillna(0).to_numpy(dtype=np.float32)
    ad_n2[b_dkey == ""] = -1
    dgroups = pd.Series(np.arange(len(s1_dkey))).groupby(s1_dkey.to_numpy()).apply(lambda x: x.to_numpy())
    ad_best = np.full(len(s1_row), np.nan, dtype=np.float32)
    ad_better = np.full(len(s1_row), np.nan, dtype=np.float32)
    ad_second = np.full(len(s1_row), np.nan, dtype=np.float32)
    s1_addr_arr = s1_addr.to_numpy()
    for i in np.flatnonzero((ad_n2 >= 1) & (ad_n2 <= 300)):
        rs = dgroups[b_dkey[i]]
        sc = np.fromiter((fuzz.token_set_ratio(b_addr[i], s1_addr_arr[r]) for r in rs), dtype=np.float32, count=len(rs))
        ad_best[i] = sc.max()
        ad_better[i] = (sc > asim[i]).sum()
        others = np.sort(sc[rs != s1_row[i]])
        ad_second[i] = others[-1] if len(others) else np.nan
    words = pd.Series(b_plain).str.split()
    return pd.DataFrame({
        "asim": asim, "ad_n2": ad_n2, "ad_gap_best": asim - ad_best, "ad_better": ad_better,
        "ad_gap_second": asim - ad_second, "dkey_eq": (s1_dkey.to_numpy()[s1_row] == b_dkey).astype(np.float32),
        "n_words": words.str.len().to_numpy(dtype=np.float32),
        "max_word": words.map(lambda w: max((len(x) for x in w), default=0)).to_numpy(dtype=np.float32),
        "empty": s23_empty[s23_row].astype(np.float32), "key_eq": (a_key == b_key).astype(np.float32),
        "n1": n1.to_numpy()[s1_row].astype(np.float32), "n2": n2, "sim": sim, "sim_key": sim_key, "tset": tset,
        "gap_best": sim - best, "n_better": n_better, "n_tie": n_tie, "gap_second": sim - second,
        "empty_x_n2": s23_empty[s23_row] * n2})


def group_features(s1_row, p):
    n = len(p)
    p = p.astype(np.float64)
    start = np.r_[True, s1_row[1:] != s1_row[:-1]]
    first = np.flatnonzero(start)
    group = np.cumsum(start) - 1
    rank = np.arange(n) - first[group]
    p1 = p[first][group]
    prev = np.where(rank > 0, np.r_[1.0, p[:-1]], 1.0)
    last = np.r_[s1_row[1:] != s1_row[:-1], True]
    nxt = np.where(~last, np.r_[p[1:], 0.0], 0.0)
    f = {"p": p, "lp": logit(p), "rank": rank, "p1": p1, "gap_p1": logit(p1) - logit(p),
         "p_prev": prev, "p_next": nxt, "size": np.bincount(group)[group]}
    for t in (0.9, 0.5, 0.1):
        f[f"n_gt{t}"] = np.bincount(group, weights=(p > t).astype(float))[group]
    return pd.DataFrame(f, dtype=np.float32), rank


PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)
ROUNDS = 600


def score(index, n_rows, keep, correct, n_true):
    n_pred = np.bincount(index[keep], minlength=n_rows).astype(float)
    n_corr = np.bincount(index[keep & correct], minlength=n_rows).astype(float)
    return f05(n_pred, n_true, n_corr)


def prepared_pairs(s1, s23, p):
    m = p > LOW
    s1, s23, p = s1[m], s23[m], p[m]
    o = exclusive_sorted(s1, s23, p)
    return s1[o], s23[o], p[o]


def main():
    pairs = pd.read_parquet(DEC / "holdout_pairs.parquet", columns=["s1_row", "s23_row", "probability"])
    truth = pd.read_parquet(DEC / "holdout_truth.parquet")
    rows = pd.read_parquet(DEC / "holdout_rows.parquet").sort_values("s1_row").reset_index(drop=True)
    s1, s23, p = prepared_pairs(*(pairs[c].to_numpy() for c in ("s1_row", "s23_row", "probability")))
    del pairs
    key = s1.astype(np.int64) * 10_000_000 + s23
    correct = np.isin(key, truth["s1_row"].to_numpy().astype(np.int64) * 10_000_000 + truth["s23_row"].to_numpy())
    rrows = rows["s1_row"].to_numpy()
    n_rows = len(rrows)
    n_true = np.bincount(np.searchsorted(rrows, truth["s1_row"].to_numpy()), minlength=n_rows).astype(float)
    index = np.searchsorted(rrows, s1)
    country = rows["country"].to_numpy()
    say(f"holdout: {n_rows:,} records, {len(p):,} exclusive pairs (p>{LOW}), {correct.sum():,} correct")

    G, rank = group_features(s1, p)
    tr_data = split_data("train")
    X = pd.concat([G, text_features(s1, s23, *tr_data)], axis=1)
    del tr_data
    window = (p > LOW) & (p < HIGH)
    say(f"re-scored window: {window.sum():,} pairs ({correct[window].sum():,} correct)")

    fold_rec = ((rrows.astype(np.uint64) * np.uint64(2654435761) % np.uint64(4294967296)) % np.uint64(2)).astype(int)
    fold = fold_rec[index]
    q = p.astype(np.float64).copy()
    for k in (0, 1):
        tr, te = window & (fold != k), window & (fold == k)
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], correct[tr].astype(int)), ROUNDS)
        q[te] = m.predict(X[te])
    top1 = score(index, n_rows, keep_rule(p, rank, 0.74, 0.56), correct, n_true)
    say(f"baseline TOP1(0.74,0.56) on p: {top1.mean():.5f}")
    order_q = np.lexsort((-q, s1))                       # rank by q within each record
    start = np.r_[True, s1[order_q][1:] != s1[order_q][:-1]]
    rank_q = np.empty(len(q), dtype=np.int64)
    rank_q[order_q] = np.arange(len(q)) - np.flatnonzero(start)[np.cumsum(start) - 1]
    results = {}
    for t in np.round(np.arange(0.40, 0.90, 0.02), 2):
        for t1 in [None] + list(np.round(np.arange(0.30, 0.70, 0.04), 2)):
            if t1 is None or t1 < t:
                results[(t, t1)] = score(index, n_rows, keep_rule(q, rank_q, t, t1), correct, n_true)
    best = max(results, key=lambda c: results[c].mean())
    say(f"ctx2 best {best}: {results[best].mean():.5f}  (gain vs TOP1 {results[best].mean() - top1.mean():+.5f})")
    honest = 0.0
    for k in (0, 1):
        pick = max(results, key=lambda c: results[c][fold_rec == k].mean())
        g = (results[pick] - top1)[fold_rec != k].mean()
        honest += g * (fold_rec != k).mean()
        say(f"  pick on half {k}: {pick} -> gain on other half {g:+.5f}")
    say(f"2-fold honest gain vs TOP1: {honest:+.5f}")
    for c in ("India", "US"):
        mk = country == c
        say(f"  {c}: TOP1 {top1[mk].mean():.5f} -> ctx2 {results[best][mk].mean():.5f}")
    for name, mk in (("0", n_true == 0), ("1", n_true == 1), ("2-3", (n_true >= 2) & (n_true <= 3)), ("4+", n_true >= 4)):
        say(f"  true matches {name}: {top1[mk].mean():.4f} -> {results[best][mk].mean():.4f} ({mk.sum():,})")
    json.dump({"best": [best[0], best[1]], "honest_gain": honest, "holdout": results[best].mean(),
               "top1": top1.mean()}, open(OUT / "ctx3_result.json", "w"))

    final = lgb.train(PARAMS, lgb.Dataset(X[window], correct[window].astype(int)), ROUNDS)
    final.save_model(str(OUT / "ctx3_model.txt"))
    imp = sorted(zip(final.feature_importance("gain"), X.columns), reverse=True)[:12]
    say("importance: " + ", ".join(f"{n} {g:.0f}" for g, n in imp))
    if "--test" not in sys.argv:
        return
    i = sys.argv.index("--test")
    apply_test(final, Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), best)


def apply_test(model, probs, out_dir, best):
    del_cols = ["s1_row", "s23_row", "probability"]
    t = pd.read_parquet(probs, columns=del_cols)
    s1, s23, p = prepared_pairs(*(t[c].to_numpy() for c in del_cols))
    del t
    say(f"test: {len(p):,} exclusive pairs (p>{LOW}) from {probs}")
    G, _ = group_features(s1, p)
    window = (p > LOW) & (p < HIGH)
    w = np.flatnonzero(window)                  # only the re-scored pairs need text features
    X = pd.concat([G.iloc[w].reset_index(drop=True), text_features(s1[w], s23[w], *split_data("test"))], axis=1)
    q = p.astype(np.float64).copy()
    q[window] = model.predict(X, num_threads=4)
    del X, G
    order_q = np.lexsort((-q, s1))
    s1, s23, q, p = s1[order_q], s23[order_q], q[order_q], p[order_q]
    start = np.r_[True, s1[1:] != s1[:-1]]
    rank_q = np.arange(len(q)) - np.flatnonzero(start)[np.cumsum(start) - 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"s1_row": s1, "s23_row": s23, "probability": q.astype(np.float32), "p_orig": p}).to_parquet(
        out_dir / "test_pair_probabilities_ctx3.parquet", index=False)
    keep = keep_rule(q, rank_q, best[0], best[1])
    s1_ids = read_cols(DATA / "test/test_source1.tsv", ["entity_id"])["entity_id"].to_numpy(zero_copy_only=False)
    s23_ids = pa.concat_arrays([read_cols(DATA / f"test/test_source{k}.tsv", ["entity_id"])["entity_id"] for k in (2, 3)])
    a, b = s1[keep], s23[keep]
    sid = np.asarray(s23_ids.take(pa.array(b)).to_numpy(zero_copy_only=False), dtype=object)
    joined = pd.Series(sid).groupby(a).agg(",".join)
    col = joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()
    pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": col}).to_csv(
        out_dir / "matching_results.tsv", sep="\t", index=False, lineterminator="\n")
    base = (p >= 0.74) | ((np.r_[True, s1[1:] != s1[:-1]]) & (p >= 0.56))
    say(f"wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs ({keep.sum() - base.sum():+,} vs TOP1 on p "
        f"in this order), {len(joined):,} records answered; changed pairs: {(keep != base).sum():,}")
    say("ALL DONE")


if __name__ == "__main__":
    main()

"""
ctx6: ctx3 + rival features on the PIPELINE's cleaned text (transliterated Indian scripts).
ctx3's plain() (NFKD -> ascii) turns Hindi-script text into "" and ~10% of Source 2/3 names and
addresses are in Indian scripts, so its rival features were blind there. Here names/addresses go
through code/normalize.py (indic_to_latin + anyascii, legal words, address short forms).
New: inside the same-name group, is the copy's ADDRESS closer to this record than to every other
same-name record (the tie-breaker for chains with one name and many addresses).
2-fold on dec1's holdout (same folds as ctx3/ctx4), honest threshold pick; compared with ctx3 alone.
Usage: python ctx/ctx6.py [--test <wide30-style probs.parquet> <out_dir>]
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "D:/AmazonML/kaggle_build/code")
from ctx1 import exclusive_sorted, f05, keep_rule  # noqa: E402
import ctx3  # noqa: E402
import ctx4  # noqa: E402
from normalize import clean_address, clean_name  # noqa: E402

KB = Path("D:/AmazonML/kaggle_build")
DEC = KB / "out_dec1/work/dec"
DATA = Path("D:/AmazonML/student_resource/dataset")
OUT = KB / "out_ctx6"
LOW, HIGH = ctx3.LOW, ctx3.HIGH
MAX_RIVALS = 300
T0 = time.time()
LOG = []
NR = ["nr_empty", "nr_n1", "nr_n2", "nr_sim", "nr_core_eq", "nr_legal_eq", "nr_gap_best", "nr_n_better",
      "nr_n_tie", "nr_gap_second", "nr_asim", "nr_na_gap", "nr_na_better", "nr_na_rank", "nr_ad_n2",
      "nr_ad_gap_best", "nr_ad_better", "nr_dkey_eq"]


def say(m):
    m = f"[{time.time() - T0:6.0f}s] {m}"
    print(m, flush=True)
    LOG.append(m)
    (OUT / "ctx6_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def _name(x):
    c, core, legal = clean_name(x or "")
    return c, core, legal


def _addr(x):
    return clean_address(x or "")


def unique_map(func, values, n_jobs=4):
    codes, uniques = pd.factorize(pd.Series(values, dtype=object).fillna(""), sort=False)
    uniques = list(uniques)
    with Pool(n_jobs) as pool:
        res = pool.map(func, uniques, chunksize=5000)
    return [res[c] for c in codes]


def read(path, cols):
    t = pc.read_csv(path, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                    convert_options=pc.ConvertOptions(include_columns=cols,
                                                      column_types={c: pa.string() for c in cols}))
    return t


def digit_key(s):
    return s.str.findall(r"\d+").map(lambda x: " ".join(sorted(set(x))))


def s1_table(split):
    cache = OUT / f"s1_norm_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    t = read(DATA / split / f"{split}_source1.tsv", ["business_name", "business_address"]).to_pandas()
    n = unique_map(_name, t["business_name"].to_numpy())
    a = unique_map(_addr, t["business_address"].to_numpy())
    d = pd.DataFrame({"name": [x[0] for x in n], "core": [x[1] for x in n], "legal": [x[2] for x in n], "addr": a})
    d["dkey"] = digit_key(d["addr"])
    d.to_parquet(cache, index=False)
    return d


def s23_rows(split, rows):
    """cleaned name/core/legal/addr of the given Source 2/3 rows (concat(source2, source3) order)."""
    parts = [read(DATA / split / f"{split}_source{k}.tsv", ["business_name", "business_address"]) for k in (2, 3)]
    names = pa.concat_arrays([p.column("business_name").combine_chunks() for p in parts])
    addrs = pa.concat_arrays([p.column("business_address").combine_chunks() for p in parts])
    take = pa.array(rows)
    n = unique_map(_name, names.take(take).to_numpy(zero_copy_only=False))
    a = unique_map(_addr, addrs.take(take).to_numpy(zero_copy_only=False))
    d = pd.DataFrame({"name": [x[0] for x in n], "core": [x[1] for x in n], "legal": [x[2] for x in n], "addr": a})
    d["dkey"] = digit_key(d["addr"])
    return d


def groups(keys):
    codes, uniques = pd.factorize(keys)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    return pd.Index(uniques), order, bounds


def nr_features(s1r, s23r, split):
    """features of the pairs (s1r, s23r) — only called for window pairs."""
    T = s1_table(split)
    u, inv = np.unique(s23r, return_inverse=True)
    B = s23_rows(split, u).iloc[inv].reset_index(drop=True)
    say(f"{split}: cleaned {len(u):,} Source 2/3 records")
    s1n, s1c, s1l, s1a, s1d = (T[c].to_numpy() for c in ("name", "core", "legal", "addr", "dkey"))
    gn, ga = groups(s1c), groups(s1d)
    kc, dc = T["core"].value_counts(), T["dkey"].value_counts()
    bn, bc, bl, ba, bd = (B[c].to_numpy() for c in ("name", "core", "legal", "addr", "dkey"))
    n = len(s1r)
    f = {c: np.full(n, np.nan, dtype=np.float32) for c in NR}
    f["nr_empty"] = (ba == "").astype(np.float32)
    f["nr_n1"] = T["core"].map(kc).to_numpy(dtype=np.float32)[s1r]
    f["nr_n2"] = pd.Series(bc).map(kc).fillna(0).to_numpy(dtype=np.float32)
    f["nr_core_eq"] = (s1c[s1r] == bc).astype(np.float32)
    f["nr_legal_eq"] = (s1l[s1r] == bl).astype(np.float32)
    f["nr_dkey_eq"] = np.where(bd == "", -1, s1d[s1r] == bd).astype(np.float32)
    ad2 = pd.Series(bd).map(dc).fillna(0).to_numpy(dtype=np.float32)
    ad2[bd == ""] = -1
    f["nr_ad_n2"] = ad2
    for i in range(n):
        x = s1r[i]
        sim = fuzz.ratio(bn[i], s1n[x])
        asim = fuzz.token_set_ratio(ba[i], s1a[x]) if ba[i] else np.nan
        f["nr_sim"][i], f["nr_asim"][i] = sim, asim
        code = gn[0].get_loc(bc[i]) if bc[i] and bc[i] in gn[0] else -1
        if code >= 0:
            rs = gn[1][gn[2][code]:gn[2][code + 1]]
            if 0 < len(rs) <= MAX_RIVALS:
                others = rs[rs != x]
                if len(others):
                    sc = np.fromiter((fuzz.ratio(bn[i], s1n[r]) for r in others), np.float32, len(others))
                    f["nr_gap_best"][i] = sim - sc.max()
                    f["nr_n_better"][i] = (sc > sim).sum()
                    f["nr_n_tie"][i] = (sc == sim).sum()
                    f["nr_gap_second"][i] = sim - np.sort(sc)[-2] if len(sc) > 1 else sim - sc.max()
                    if ba[i]:
                        asc = np.fromiter((fuzz.token_set_ratio(ba[i], s1a[r]) for r in others), np.float32,
                                          len(others))
                        f["nr_na_gap"][i] = asim - asc.max()
                        f["nr_na_better"][i] = (asc > asim).sum()
                        f["nr_na_rank"][i] = (asc >= asim).sum()
        if bd[i] and bd[i] in ga[0]:
            code = ga[0].get_loc(bd[i])
            rs = ga[1][ga[2][code]:ga[2][code + 1]]
            others = rs[rs != x]
            if 0 < len(others) <= MAX_RIVALS and ba[i]:
                sc = np.fromiter((fuzz.token_set_ratio(ba[i], s1a[r]) for r in others), np.float32, len(others))
                f["nr_ad_gap_best"][i] = asim - sc.max()
                f["nr_ad_better"][i] = (sc > asim).sum()
        if i and i % 200000 == 0:
            say(f"  {i:,} / {n:,} pairs")
    return pd.DataFrame(f)


def main():
    OUT.mkdir(exist_ok=True)
    pairs = pd.read_parquet(DEC / "holdout_pairs.parquet", columns=["s1_row", "s23_row", "probability"])
    truth = pd.read_parquet(DEC / "holdout_truth.parquet")
    rows = pd.read_parquet(DEC / "holdout_rows.parquet").sort_values("s1_row").reset_index(drop=True)
    s1, s23, p = (pairs[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del pairs
    o = exclusive_sorted(s1, s23, p)
    s1, s23, p = s1[o], s23[o], p[o].astype(np.float64)
    tkey = truth["s1_row"].to_numpy().astype(np.int64) * 10_000_000 + truth["s23_row"].to_numpy()
    correct = np.isin(s1.astype(np.int64) * 10_000_000 + s23, tkey)
    rrows = rows["s1_row"].to_numpy()
    n_rows = len(rrows)
    n_true = np.bincount(np.searchsorted(rrows, truth["s1_row"].to_numpy()), minlength=n_rows).astype(float)
    index = np.searchsorted(rrows, s1)
    fold_rec = ((rrows.astype(np.uint64) * np.uint64(2654435761) % np.uint64(4294967296)) % np.uint64(2)).astype(int)
    fold = fold_rec[index]
    X3, w = ctx4.features(s1, s23, p, "train")
    say(f"ctx3 features: {X3.shape}")
    XN = nr_features(s1[w], s23[w], "train")
    X6 = pd.concat([X3, XN], axis=1)
    y = correct[w].astype(int)
    rank0, _, _ = ctx4.rank_in_group(s1)

    def sc(qq):
        o2 = np.lexsort((-qq, s1))
        r = np.empty(len(qq), dtype=np.int64)
        r[o2] = ctx4.rank_in_group(s1[o2])[0]
        res = {}
        for t in np.round(np.arange(0.50, 0.90, 0.02), 2):
            for t1 in [None] + list(np.round(np.arange(0.40, 0.74, 0.04), 2)):
                if t1 is not None and t1 >= t:
                    continue
                k = keep_rule(qq, r, t, t1)
                n_pred = np.bincount(index[k], minlength=n_rows).astype(float)
                n_corr = np.bincount(index[k & correct], minlength=n_rows).astype(float)
                res[(t, t1)] = f05(n_pred, n_true, n_corr)
        return res
    k0 = keep_rule(p, rank0, 0.74, 0.56)
    base = f05(np.bincount(index[k0], minlength=n_rows).astype(float), n_true,
               np.bincount(index[k0 & correct], minlength=n_rows).astype(float))
    out = {}
    for name, X in (("ctx3", X3), ("ctx6", X6)):
        q = p.copy()
        for k in (0, 1):
            tr, te = fold[w] != k, fold[w] == k
            m = lgb.train(ctx4.PARAMS, lgb.Dataset(X[tr], y[tr]), ctx4.ROUNDS)
            q[w[te]] = m.predict(X[te])
        res = sc(q)
        best = max(res, key=lambda c: res[c].mean())
        honest = 0.0
        for k in (0, 1):
            pick = max(res, key=lambda c: res[c][fold_rec == k].mean())
            honest += (res[pick] - base)[fold_rec != k].mean() * (fold_rec != k).mean()
        country = rows["country"].to_numpy()
        byc = {c: float((res[best] - base)[country == c].mean()) for c in ("India", "US")}
        say(f"{name}: best {best} {res[best].mean():.5f}  honest gain vs TOP1 {honest:+.5f}  {byc}")
        out[name] = (best, honest, byc)
    passes = out["ctx6"][1] > out["ctx3"][1] + 0.0002
    say("ctx6 PASSES (beats ctx3)" if passes else "ctx6 does NOT beat ctx3")
    best = out["ctx6"][0]
    json.dump({"best": [best[0], best[1]], "honest_ctx6": out["ctx6"][1], "honest_ctx3": out["ctx3"][1],
               "passes": bool(passes)}, open(OUT / "ctx6_result.json", "w"))
    final = lgb.train(ctx4.PARAMS, lgb.Dataset(X6, y), ctx4.ROUNDS)
    imp = sorted(zip(final.feature_importance("gain"), X6.columns), reverse=True)[:12]
    say("importance: " + ", ".join(f"{n} {g:.0f}" for g, n in imp))
    if "--test" in sys.argv and passes:
        i = sys.argv.index("--test")
        apply_test(final, Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), best)


def apply_test(model, probs, out_dir, best):
    d = pd.read_parquet(probs, columns=["s1_row", "s23_row", "probability"])
    s1, s23, p = (d[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del d
    o = exclusive_sorted(s1, s23, p)
    s1, s23, p = s1[o], s23[o], p[o].astype(np.float64)
    del o
    X3, w = ctx4.features(s1, s23, p, "test")
    XN = nr_features(s1[w], s23[w], "test")
    q = p.copy()
    q[w] = model.predict(pd.concat([X3, XN], axis=1), num_threads=4)
    del X3, XN
    o = np.lexsort((-q, s1))
    s1, s23, q, p = s1[o], s23[o], q[o], p[o]
    rank, _, _ = ctx4.rank_in_group(s1)
    keep = keep_rule(q, rank, best[0], best[1])

    def ids(name):
        t = read(DATA / "test" / name, ["entity_id"])
        return t.column("entity_id").combine_chunks()
    s1_ids = ids("test_source1.tsv").to_numpy(zero_copy_only=False)
    s23_ids = pa.concat_arrays([ids("test_source2.tsv"), ids("test_source3.tsv")])
    a, b = s1[keep], s23[keep]
    sid = np.asarray(s23_ids.take(pa.array(b)).to_numpy(zero_copy_only=False), dtype=object)
    joined = pd.Series(sid).groupby(a).agg(",".join)
    col = joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": col}).to_csv(
        out_dir / "matching_results.tsv", sep="\t", index=False, lineterminator="\n")
    pd.DataFrame({"s1_row": s1, "s23_row": s23, "probability": q.astype(np.float32),
                  "p_orig": p.astype(np.float32)}).to_parquet(out_dir / "test_pair_probabilities_ctx6.parquet",
                                                              index=False)
    say(f"test: wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs, {len(joined):,} records answered")


if __name__ == "__main__":
    main()

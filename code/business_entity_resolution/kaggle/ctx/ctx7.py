"""
ctx7: ctx6 + WORD-DIFFERENCE features (decoys vs true copies).
France (15% of the test, no labels) has 3x more unsure pairs than India/US. Looking at them: the
unsure negatives are decoys at the SAME address whose name got words added or swapped ("& Fils",
"Groupe", "Comite", "Developpement"); true copies change the legal form, use initials ("FC") or a
domain name. The features below say that without naming any language's words, so what is learned on
India/US decoys ("& Sons", "Group") can carry over.
Usage: python ctx/ctx7.py [--test <wide30-style probs.parquet> <out_dir>]
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import ctx6  # noqa: E402
from ctx6 import (DEC, OUT as _O, exclusive_sorted, f05, keep_rule, ctx4, nr_features,  # noqa: E402,F401
                  s1_table, s23_rows)

OUT = Path("D:/AmazonML/kaggle_build/out_ctx7")
ctx6.OUT = OUT          # say() logs + caches go here (s1 caches copied below)
WD = ["wd_add", "wd_miss", "wd_common", "wd_add_short", "wd_add_long", "wd_first_eq", "wd_order_eq",
      "wd_acronym", "wd_concat", "wd_legal_diff", "wd_legal_b_only", "wd_legal_x_only", "wd_jacc",
      "wd_b_ntok", "wd_x_ntok", "wd_rank_add"]


def acronym(tokens):
    return "".join(t[0] for t in tokens if t)


def wd_features(s1r, s23r, split, p=None):
    T = s1_table(split)
    u, inv = np.unique(s23r, return_inverse=True)
    B = s23_rows(split, u).iloc[inv].reset_index(drop=True)
    xc, xl, xn = T["core"].to_numpy()[s1r], T["legal"].to_numpy()[s1r], T["name"].to_numpy()[s1r]
    bc, bl = B["core"].to_numpy(), B["legal"].to_numpy()
    n = len(s1r)
    f = {c: np.zeros(n, dtype=np.float32) for c in WD}
    for i in range(n):
        tx, tb = (xc[i] or "").split(), (bc[i] or "").split()
        sx, sb = set(tx), set(tb)
        add, miss = sb - sx, sx - sb
        f["wd_add"][i], f["wd_miss"][i], f["wd_common"][i] = len(add), len(miss), len(sx & sb)
        f["wd_add_short"][i] = sum(len(t) <= 3 for t in add)
        f["wd_add_long"][i] = sum(len(t) > 3 for t in add)
        f["wd_first_eq"][i] = float(bool(tx) and bool(tb) and tx[0] == tb[0])
        common = [t for t in tb if t in sx]
        f["wd_order_eq"][i] = float(common == [t for t in tx if t in sb])
        jb = "".join(tb)
        f["wd_acronym"][i] = float(len(tb) == 1 and len(tx) > 1 and (jb == acronym(tx) or jb == acronym((xn[i] or "").split())))
        f["wd_concat"][i] = float(len(tb) == 1 and len(tx) > 1 and (jb == "".join(tx) or jb == (xn[i] or "").replace(" ", "")))
        f["wd_legal_diff"][i] = float(bool(xl[i]) and bool(bl[i]) and xl[i] != bl[i])
        f["wd_legal_b_only"][i] = float(bool(bl[i]) and not xl[i])
        f["wd_legal_x_only"][i] = float(bool(xl[i]) and not bl[i])
        f["wd_jacc"][i] = len(sx & sb) / max(len(sx | sb), 1)
        f["wd_b_ntok"][i], f["wd_x_ntok"][i] = len(tb), len(tx)
    # within the record: how many of its candidates add FEWER words (decoys add the most)
    d = pd.DataFrame({"s": s1r, "a": f["wd_add"]})
    f["wd_rank_add"] = d.groupby("s")["a"].rank(method="min").to_numpy(dtype=np.float32)
    return pd.DataFrame(f)


def main():
    OUT.mkdir(exist_ok=True)
    for split in ("train", "test"):
        src, dst = _O / f"s1_norm_{split}.parquet", OUT / f"s1_norm_{split}.parquet"
        if src.exists() and not dst.exists():
            dst.write_bytes(src.read_bytes())
    say = ctx6.say
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
    XN = nr_features(s1[w], s23[w], "train")
    XW = wd_features(s1[w], s23[w], "train")
    say(f"features: ctx3 {X3.shape[1]}, nr {XN.shape[1]}, wd {XW.shape[1]} on {len(w):,} pairs")
    y = correct[w].astype(int)
    rank0, _, _ = ctx4.rank_in_group(s1)
    k0 = keep_rule(p, rank0, 0.74, 0.56)
    base = f05(np.bincount(index[k0], minlength=n_rows).astype(float), n_true,
               np.bincount(index[k0 & correct], minlength=n_rows).astype(float))
    country = rows["country"].to_numpy()

    def grid(qq):
        o2 = np.lexsort((-qq, s1))
        r = np.empty(len(qq), dtype=np.int64)
        r[o2] = ctx4.rank_in_group(s1[o2])[0]
        res = {}
        for t in np.round(np.arange(0.50, 0.90, 0.02), 2):
            for t1 in [None] + list(np.round(np.arange(0.40, 0.74, 0.04), 2)):
                if t1 is not None and t1 >= t:
                    continue
                k = keep_rule(qq, r, t, t1)
                res[(t, t1)] = f05(np.bincount(index[k], minlength=n_rows).astype(float), n_true,
                                   np.bincount(index[k & correct], minlength=n_rows).astype(float))
        return res
    out = {}
    for name, X in (("ctx3", X3), ("ctx6", pd.concat([X3, XN], axis=1)), ("ctx7", pd.concat([X3, XN, XW], axis=1))):
        q = p.copy()
        for k in (0, 1):
            tr, te = fold[w] != k, fold[w] == k
            m = lgb.train(ctx4.PARAMS, lgb.Dataset(X[tr], y[tr]), ctx4.ROUNDS)
            q[w[te]] = m.predict(X[te])
        res = grid(q)
        best = max(res, key=lambda c: res[c].mean())
        honest = 0.0
        for k in (0, 1):
            pick = max(res, key=lambda c: res[c][fold_rec == k].mean())
            honest += (res[pick] - base)[fold_rec != k].mean() * (fold_rec != k).mean()
        byc = {c: round(float((res[best] - base)[country == c].mean()), 5) for c in ("India", "US")}
        say(f"{name}: best {best} {res[best].mean():.5f}  honest gain vs TOP1 {honest:+.5f}  {byc}")
        out[name] = (best, honest, X)
    passes = out["ctx7"][1] > out["ctx3"][1] + 0.0002
    say("ctx7 PASSES (beats ctx3 by > 0.0002)" if passes else "ctx7 does NOT beat ctx3 by 0.0002")
    best, _, X7 = out["ctx7"]
    json.dump({"best": [best[0], best[1]], **{f"honest_{k}": v[1] for k, v in out.items()}, "passes": bool(passes)},
              open(OUT / "ctx7_result.json", "w"))
    final = lgb.train(ctx4.PARAMS, lgb.Dataset(X7, y), ctx4.ROUNDS)
    imp = sorted(zip(final.feature_importance("gain"), X7.columns), reverse=True)[:14]
    say("importance: " + ", ".join(f"{n} {g:.0f}" for g, n in imp))
    if "--test" in sys.argv and (passes or "--force" in sys.argv):
        i = sys.argv.index("--test")
        apply_test(final, Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), best)


def apply_test(model, probs, out_dir, best):
    import pyarrow as pa
    say = ctx6.say
    d = pd.read_parquet(probs, columns=["s1_row", "s23_row", "probability"])
    s1, s23, p = (d[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del d
    o = exclusive_sorted(s1, s23, p)
    s1, s23, p = s1[o], s23[o], p[o].astype(np.float64)
    del o
    X3, w = ctx4.features(s1, s23, p, "test")
    XN = nr_features(s1[w], s23[w], "test")
    XW = wd_features(s1[w], s23[w], "test")
    q = p.copy()
    q[w] = model.predict(pd.concat([X3, XN, XW], axis=1), num_threads=4)
    del X3, XN, XW
    o = np.lexsort((-q, s1))
    s1, s23, q, p = s1[o], s23[o], q[o], p[o]
    rank, _, _ = ctx4.rank_in_group(s1)
    keep = keep_rule(q, rank, best[0], best[1])

    def ids(name):
        return ctx6.read(ctx6.DATA / "test" / name, ["entity_id"]).column("entity_id").combine_chunks()
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
                  "p_orig": p.astype(np.float32)}).to_parquet(out_dir / "test_pair_probabilities_ctx7.parquet",
                                                              index=False)
    say(f"test: wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs, {len(joined):,} records answered")


if __name__ == "__main__":
    main()

"""
ctx5: SIBLING EXPANSION for blocking misses (on top of ctx3 + threshold decoding).
7% of holdout records have true links that blocking never proposed, mostly big clusters (3-6 copies).
A missed copy B usually carries the same name as a copy A that WAS found (only the address was
re-written), so: for every kept pair (X, A), propose the Source 2/3 records B with the same name key as A
that are not already owned by X, score (X, B) with a small model, and add those above a threshold.
Trained 2-fold on dec1's holdout (honest threshold pick on halves), applied to a ctx3 test output.
Usage: python ctx/ctx5.py [--test <ctx3 probs.parquet> <out_dir>]
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
from ctx1 import exclusive_sorted, f05, keep_rule  # noqa: E402
import ctx3  # noqa: E402
import ctx4  # noqa: E402

KB = Path("D:/AmazonML/kaggle_build")
DEC = KB / "out_dec1/work/dec"
OUT = KB / "out_ctx5"
SEED_Q = 0.5
MAX_GROUP = 30
OWNED = 0.3          # B already exclusively kept by another record with q >= this: not proposed
T0 = time.time()
LOG = []


def say(m):
    m = f"[{time.time() - T0:6.0f}s] {m}"
    print(m, flush=True)
    LOG.append(m)
    (OUT / "ctx5_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def s23_keys(split, names):
    cache = OUT / f"s23_keys_{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["key"].to_numpy()
    keys = []
    for s in range(0, len(names), 1_000_000):
        keys.append(ctx3.name_key(ctx3.plain(names.slice(s, 1_000_000).to_numpy(zero_copy_only=False))).to_numpy())
    keys = np.concatenate(keys)
    pd.DataFrame({"key": keys}).to_parquet(cache, index=False)
    return keys


def expansions(s1, s23, q, keep, split):
    """Candidate (X, B) pairs + features. s1/s23/q: exclusive pairs; keep: kept mask."""
    s1_plain, s1_key, s23_names, s23_empty, s1_addr, s23_addrs = ctx3.split_data(split)
    s1_plain, s1_key, s1_addr = (np.asarray(x, dtype=object) for x in (s1_plain, s1_key, s1_addr))
    s23_names, s23_addrs = (x if isinstance(x, pa.Array) else pa.array(np.asarray(x, dtype=object), pa.string())
                            for x in (s23_names, s23_addrs))
    keys = s23_keys(split, s23_names)
    say(f"{split}: {len(keys):,} Source 2/3 keys")
    codes, uniques = pd.factorize(keys)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(uniques) + 1))
    gsize = np.diff(bounds)
    s1_kc = pd.Series(s1_key).value_counts()
    owner = np.full(len(keys), -1, dtype=np.int64)          # exclusive owner (kept) of each s23 record
    owner_q = np.zeros(len(keys), dtype=np.float32)
    owner[s23[keep]] = s1[keep]
    owner_q[s23[keep]] = q[keep]
    seed = keep & (q >= SEED_Q)
    sx, sa, sq = s1[seed], s23[seed], q[seed]
    sc = codes[sa]
    ok = (sc >= 0) & (gsize[np.maximum(sc, 0)] <= MAX_GROUP) & (keys[sa] != "")
    sx, sa, sq, sc = sx[ok], sa[ok], sq[ok], sc[ok]
    lens = gsize[sc]
    X = np.repeat(sx, lens)
    A = np.repeat(sa, lens)
    Q = np.repeat(sq, lens)
    starts = np.repeat(bounds[sc], lens)
    off = np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens)
    B = order[starts + off]
    m = (B != A) & (owner[B] != X) & ~((owner[B] >= 0) & (owner_q[B] >= OWNED))
    X, A, B, Q = X[m], A[m], B[m], Q[m]
    df = pd.DataFrame({"X": X, "A": A, "B": B, "seed_q": Q})
    agg = df.groupby(["X", "B"], sort=False).agg(seed_q=("seed_q", "max"), n_seeds=("A", "size"),
                                                 A=("A", "first")).reset_index()
    X, B, A = agg["X"].to_numpy(), agg["B"].to_numpy(), agg["A"].to_numpy()
    say(f"{split}: {len(agg):,} expansion candidates from {seed.sum():,} seeds")
    take = pa.array(np.unique(np.r_[A, B]))
    uniq = take.to_numpy()
    nm = dict(zip(uniq, ctx3.plain(s23_names.take(take).to_numpy(zero_copy_only=False))))
    ad = dict(zip(uniq, ctx3.plain(s23_addrs.take(take).to_numpy(zero_copy_only=False))))
    bn = np.array([nm[b] for b in B], dtype=object)
    an = np.array([nm[a] for a in A], dtype=object)
    ba = np.array([ad[b] for b in B], dtype=object)
    aa = np.array([ad[a] for a in A], dtype=object)
    xn = ctx3.plain(s1_plain[X]).to_numpy()
    xa = ctx3.plain(s1_addr[X]).to_numpy()
    n_kept = np.bincount(s1[keep], minlength=len(s1_plain))
    F = pd.DataFrame({
        "seed_q": agg["seed_q"].to_numpy(dtype=np.float32), "n_seeds": agg["n_seeds"].to_numpy(dtype=np.float32),
        "gsize": gsize[codes[B]].astype(np.float32),
        "s1_rivals": pd.Series(keys[B]).map(s1_kc).fillna(0).to_numpy(dtype=np.float32),
        "key_eq_x": (s1_key[X] == keys[B]).astype(np.float32),
        "sim_bx": np.array([fuzz.ratio(a, b) for a, b in zip(bn, xn)], dtype=np.float32),
        "sim_ba": np.array([fuzz.ratio(a, b) for a, b in zip(bn, an)], dtype=np.float32),
        "asim_bx": np.array([fuzz.token_set_ratio(a, b) for a, b in zip(ba, xa)], dtype=np.float32),
        "asim_ba": np.array([fuzz.token_set_ratio(a, b) for a, b in zip(ba, aa)], dtype=np.float32),
        "dkey_bx": np.array([float(a == b) if a else -1.0 for a, b in
                             zip(ctx3.digit_key(pd.Series(ba)), ctx3.digit_key(pd.Series(xa)))], dtype=np.float32),
        "b_empty": (ba == "").astype(np.float32),
        "b_owned_q": np.where(owner[B] >= 0, owner_q[B], 0).astype(np.float32),
        "x_kept": n_kept[X].astype(np.float32)})
    return X, B, F


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
    X, w = ctx4.features(s1, s23, p, "train")
    q = p.copy()
    for k in (0, 1):
        tr, te = fold[w] != k, fold[w] == k
        m = lgb.train(ctx4.PARAMS, lgb.Dataset(X[tr], correct[w][tr].astype(int)), ctx4.ROUNDS)
        q[w[te]] = m.predict(X[te])
    del X
    o = np.lexsort((-q, s1))
    s1, s23, q, correct, index = s1[o], s23[o], q[o], correct[o], index[o]
    rank, _, _ = ctx4.rank_in_group(s1)
    t, t1 = 0.78, 0.66     # ctx q threshold chosen by ctx4 on this holdout
    keep = keep_rule(q, rank, t, t1)
    n_pred0 = np.bincount(index[keep], minlength=n_rows).astype(float)
    n_corr0 = np.bincount(index[keep & correct], minlength=n_rows).astype(float)
    base = f05(n_pred0, n_true, n_corr0)
    say(f"holdout ctx3 threshold ({t},{t1}): {base.mean():.5f}")

    EX, EB, EF = expansions(s1, s23, q, keep, "train")
    lab = np.isin(EX.astype(np.int64) * 10_000_000 + EB, tkey)
    eidx = np.searchsorted(rrows, EX)
    missed_total = n_true.sum() - n_corr0.sum()
    say(f"expansions: {len(EX):,}, true {lab.sum():,} (of {missed_total:,.0f} true links not kept)")
    efold = fold_rec[eidx]
    eq = np.zeros(len(EX))
    P = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.9,
             bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4, seed=7)
    for k in (0, 1):
        tr = efold != k
        m = lgb.train(P, lgb.Dataset(EF[tr], lab[tr].astype(int)), 400)
        eq[~tr] = m.predict(EF[~tr])
    # one owner per B
    ord_ = np.lexsort((-eq, EB))
    firstB = np.r_[True, EB[ord_][1:] != EB[ord_][:-1]]
    best_for_B = np.zeros(len(EX), dtype=bool)
    best_for_B[ord_[firstB]] = True

    def gain(th):
        add = best_for_B & (eq >= th)
        n_pred = n_pred0 + np.bincount(eidx[add], minlength=n_rows)
        n_corr = n_corr0 + np.bincount(eidx[add & lab], minlength=n_rows)
        return f05(n_pred, n_true, n_corr) - base
    res = {th: gain(th) for th in np.round(np.arange(0.30, 0.96, 0.05), 2)}
    for th, g in res.items():
        say(f"  add eq >= {th}: {g.mean():+.5f}  ({(best_for_B & (eq >= th)).sum():,} added)")
    honest = 0.0
    for k in (0, 1):
        pick = max(res, key=lambda c: res[c][fold_rec == k].mean())
        g = res[pick][fold_rec != k].mean()
        honest += g * (fold_rec != k).mean()
        say(f"  pick on half {k}: {pick} -> {g:+.5f} on other half")
    best = max(res, key=lambda c: res[c].mean())
    country = rows["country"].to_numpy()
    for c in ("India", "US"):
        say(f"  {c}: {res[best][country == c].mean():+.5f}")
    passes = honest > 0.0002 and all(res[best][country == c].mean() > 0 for c in ("India", "US"))
    say(f"honest expansion gain: {honest:+.5f}  -> {'PASSES' if passes else 'does NOT pass'}")
    final = lgb.train(P, lgb.Dataset(EF, lab.astype(int)), 400)
    imp = sorted(zip(final.feature_importance("gain"), EF.columns), reverse=True)[:8]
    say("importance: " + ", ".join(f"{n} {g:.0f}" for g, n in imp))
    json.dump({"best": float(best), "honest": honest, "passes": bool(passes), "t": t, "t1": t1},
              open(OUT / "ctx5_result.json", "w"))
    if "--test" in sys.argv and passes:
        i = sys.argv.index("--test")
        apply_test(final, Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), float(best), t, t1)


def apply_test(model, probs, out_dir, th, t, t1):
    import pyarrow.csv as pc
    d = pd.read_parquet(probs, columns=["s1_row", "s23_row", "probability"])
    s1, s23, q = (d[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del d
    o = exclusive_sorted(s1, s23, q)
    s1, s23, q = s1[o], s23[o], q[o].astype(np.float64)
    rank, _, _ = ctx4.rank_in_group(s1)
    keep = keep_rule(q, rank, t, t1)
    EX, EB, EF = expansions(s1, s23, q, keep, "test")
    eq = model.predict(EF, num_threads=4)
    ord_ = np.lexsort((-eq, EB))
    firstB = np.r_[True, EB[ord_][1:] != EB[ord_][:-1]]
    best_for_B = np.zeros(len(EX), dtype=bool)
    best_for_B[ord_[firstB]] = True
    add = best_for_B & (eq >= th)
    say(f"test: {keep.sum():,} kept + {add.sum():,} expansions (of {len(EX):,} candidates)")
    a = np.r_[s1[keep], EX[add]]
    b = np.r_[s23[keep], EB[add]]
    raw = ctx3.DATA / "test"

    def ids(name):
        tt = pc.read_csv(raw / name, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                         convert_options=pc.ConvertOptions(include_columns=["entity_id"],
                                                           column_types={"entity_id": pa.string()}))
        return tt.column("entity_id").combine_chunks()
    s1_ids = ids("test_source1.tsv").to_numpy(zero_copy_only=False)
    s23_ids = pa.concat_arrays([ids("test_source2.tsv"), ids("test_source3.tsv")])
    sid = np.asarray(s23_ids.take(pa.array(b)).to_numpy(zero_copy_only=False), dtype=object)
    joined = pd.Series(sid).groupby(a).agg(",".join)
    col = joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": col}).to_csv(
        out_dir / "matching_results.tsv", sep="\t", index=False, lineterminator="\n")
    pd.DataFrame({"s1_row": EX[add], "s23_row": EB[add], "eq": eq[add].astype(np.float32)}).to_parquet(
        out_dir / "expansions.parquet", index=False)
    say(f"test: wrote {out_dir / 'matching_results.tsv'} ({len(joined):,} records answered)")


if __name__ == "__main__":
    main()

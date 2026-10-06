"""
ctx4: EXPECTED-F0.5 decoder on top of the ctx3 re-scorer.
The threshold rule (keep q >= t, or rank 0 with q >= t1) treats every record alike. With calibrated
probabilities we can instead pick, per Source 1 record, the number k of top candidates that maximises
the EXPECTED F0.5 of that record (Monte Carlo over the candidates' Bernoulli outcomes, plus a small
chance r of one true link outside the candidate list). Knobs: r, a temperature a on logit(q).
Honest check: 2-fold cross-fitted q on dec1's holdout, knobs picked on one half, measured on the other.
Usage: python ctx/ctx4.py [--test <probs.parquet> <out_dir>]
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ctx1 import exclusive_sorted, f05, keep_rule, logit  # noqa: E402
import ctx3  # noqa: E402

KB = Path("D:/AmazonML/kaggle_build")
DEC = KB / "out_dec1/work/dec"
OUT = KB / "out_ctx4"
LOW, HIGH = ctx3.LOW, ctx3.HIGH
TOPK = 8
SAMPLES = 256
T0 = time.time()
LOG = []


def say(m):
    m = f"[{time.time() - T0:6.0f}s] {m}"
    print(m, flush=True)
    LOG.append(m)
    (OUT / "ctx4_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def features(s1, s23, p, split):
    G, _ = ctx3.group_features(s1, p)
    w = np.flatnonzero((p > LOW) & (p < HIGH))
    X = pd.concat([G.iloc[w].reset_index(drop=True),
                   ctx3.text_features(s1[w], s23[w], *ctx3.split_data(split))], axis=1)
    return X, w


def rank_in_group(s1):
    start = np.r_[True, s1[1:] != s1[:-1]]
    first = np.flatnonzero(start)
    group = np.cumsum(start) - 1
    return np.arange(len(s1)) - first[group], group, first


def matrix(q, rank, group, n_groups):
    """(n_groups, TOPK) probabilities of the top candidates (pairs are sorted by q within a record)."""
    M = np.zeros((n_groups, TOPK), dtype=np.float32)
    m = rank < TOPK
    M[group[m], rank[m]] = q[m]
    return M


def best_k(M, r, seed=0, chunk=20000):
    """Per record: k in 0..TOPK maximising E[F0.5] when keeping the top k."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(M), dtype=np.int16)
    ks = np.arange(TOPK + 1, dtype=np.float32)
    for s in range(0, len(M), chunk):
        P = M[s:s + chunk]
        X = rng.random((SAMPLES, len(P), TOPK), dtype=np.float32) < P[None]
        miss = (rng.random((SAMPLES, len(P)), dtype=np.float32) < r).astype(np.float32)
        T = X.sum(2, dtype=np.float32) + miss                       # true links of the record
        tp = np.concatenate([np.zeros((SAMPLES, len(P), 1), np.float32),
                             np.cumsum(X, 2, dtype=np.float32)], 2)   # TP when keeping top k
        den = 0.25 * T[..., None] + ks
        F = np.where(den > 0, 1.25 * tp / np.maximum(den, 1e-9), 1.0)
        out[s:s + chunk] = F.mean(0).argmax(1)
    return out


def temper(q, a):
    return 1 / (1 + np.exp(-a * logit(q)))


PARAMS = dict(ctx3.PARAMS)
ROUNDS = ctx3.ROUNDS


def main():
    OUT.mkdir(exist_ok=True)
    pairs = pd.read_parquet(DEC / "holdout_pairs.parquet", columns=["s1_row", "s23_row", "probability"])
    truth = pd.read_parquet(DEC / "holdout_truth.parquet")
    rows = pd.read_parquet(DEC / "holdout_rows.parquet").sort_values("s1_row").reset_index(drop=True)
    s1, s23, p = (pairs[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    o = exclusive_sorted(s1, s23, p)
    s1, s23, p = s1[o], s23[o], p[o].astype(np.float64)
    key = s1.astype(np.int64) * 10_000_000 + s23
    tkey = truth["s1_row"].to_numpy().astype(np.int64) * 10_000_000 + truth["s23_row"].to_numpy()
    correct = np.isin(key, tkey)
    rrows = rows["s1_row"].to_numpy()
    n_rows = len(rrows)
    n_true = np.bincount(np.searchsorted(rrows, truth["s1_row"].to_numpy()), minlength=n_rows).astype(float)
    index = np.searchsorted(rrows, s1)
    missed = n_true.sum() - correct.sum()
    say(f"holdout: {n_rows:,} records, {len(p):,} pairs, {correct.sum():,} correct, {missed:,.0f} true links outside")

    X, w = features(s1, s23, p, "train")
    fold_rec = ((rrows.astype(np.uint64) * np.uint64(2654435761) % np.uint64(4294967296)) % np.uint64(2)).astype(int)
    fold = fold_rec[index]
    q = p.copy()
    for k in (0, 1):
        tr, te = fold[w] != k, fold[w] == k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], correct[w][tr].astype(int)), ROUNDS)
        q[w[te]] = m.predict(X[te])
    del X
    # re-sort by q within each record
    o = np.lexsort((-q, s1))
    s1, s23, p, q, correct, index = s1[o], s23[o], p[o], q[o], correct[o], index[o]
    rank, group, first = rank_in_group(s1)
    grec = index[first]                                   # holdout record of each group

    def sc(keep):
        n_pred = np.bincount(index[keep], minlength=n_rows).astype(float)
        n_corr = np.bincount(index[keep & correct], minlength=n_rows).astype(float)
        return f05(n_pred, n_true, n_corr)

    base = sc(keep_rule(p, rank, 0.74, 0.56))
    thr = {(t, t1): sc(keep_rule(q, rank, t, t1)) for t in np.round(np.arange(0.40, 0.90, 0.02), 2)
           for t1 in [None] + list(np.round(np.arange(0.30, 0.70, 0.04), 2)) if t1 is None or t1 < t}
    bt = max(thr, key=lambda c: thr[c].mean())
    say(f"base p TOP1: {base.mean():.5f}   ctx q threshold best {bt}: {thr[bt].mean():.5f}")
    ef = {}
    for a in (0.8, 1.0, 1.2, 1.5):
        M = matrix(temper(q, a), rank, group, len(first))
        for r in (0.0, 0.01, 0.02, 0.04):
            k = best_k(M, r)
            ef[(a, r)] = sc(rank < k[group])
            say(f"  expF a={a} r={r}: {ef[(a, r)].mean():.5f}")
    be = max(ef, key=lambda c: ef[c].mean())
    say(f"expected-F best {be}: {ef[be].mean():.5f}  vs threshold {thr[bt].mean():.5f}")
    honest_t = honest_e = 0.0
    for k in (0, 1):
        other = fold_rec != k
        pt = max(thr, key=lambda c: thr[c][fold_rec == k].mean())
        pe = max(ef, key=lambda c: ef[c][fold_rec == k].mean())
        honest_t += (thr[pt] - base)[other].mean() * other.mean()
        honest_e += (ef[pe] - base)[other].mean() * other.mean()
        say(f"  half {k}: threshold {pt} / expF {pe}")
    say(f"honest gain vs p TOP1: threshold {honest_t:+.5f}   expected-F {honest_e:+.5f}")
    country = rows["country"].to_numpy()
    for c in ("India", "US"):
        mk = country == c
        say(f"  {c}: base {base[mk].mean():.5f} thr {thr[bt][mk].mean():.5f} expF {ef[be][mk].mean():.5f}")
    passes = honest_e > honest_t + 0.0002
    say("expected-F PASSES" if passes else "expected-F does NOT beat the threshold rule")
    json.dump({"thr": [bt[0], bt[1]], "expF": list(be), "honest_thr": honest_t, "honest_expF": honest_e,
               "passes": bool(passes)}, open(OUT / "ctx4_result.json", "w"))
    if "--test" in sys.argv and passes:
        i = sys.argv.index("--test")
        apply_test(Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), be)


def apply_test(probs, out_dir, knobs):
    """probs: a ctx3 test output (s1_row, s23_row, probability=q)."""
    import pyarrow as pa
    import pyarrow.csv as pc
    t = pd.read_parquet(probs, columns=["s1_row", "s23_row", "probability"])
    s1, s23, q = (t[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del t
    o = exclusive_sorted(s1, s23, q)
    s1, s23, q = s1[o], s23[o], q[o].astype(np.float64)
    rank, group, first = rank_in_group(s1)
    k = best_k(matrix(temper(q, knobs[0]), rank, group, len(first)), knobs[1])
    keep = rank < k[group]
    raw = ctx3.DATA / "test"

    def ids(name):
        tt = pc.read_csv(raw / name, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                         convert_options=pc.ConvertOptions(include_columns=["entity_id"],
                                                           column_types={"entity_id": pa.string()}))
        return tt.column("entity_id").combine_chunks()
    s1_ids = ids("test_source1.tsv").to_numpy(zero_copy_only=False)
    s23_ids = pa.concat_arrays([ids("test_source2.tsv"), ids("test_source3.tsv")])
    a, b = s1[keep], s23[keep]
    sid = np.asarray(s23_ids.take(pa.array(b)).to_numpy(zero_copy_only=False), dtype=object)
    joined = pd.Series(sid).groupby(a).agg(",".join)
    col = joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": col}).to_csv(
        out_dir / "matching_results.tsv", sep="\t", index=False, lineterminator="\n")
    say(f"test: wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs, {len(joined):,} records answered")


if __name__ == "__main__":
    main()

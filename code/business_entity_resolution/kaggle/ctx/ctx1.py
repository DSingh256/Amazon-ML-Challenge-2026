"""
ctx1: learned GROUP-CONTEXT re-scorer (local, CPU). A pair's probability p is judged alone; this model
sees the whole candidate list of its Source 1 record (rank, gap to the best / next candidate, how many
strong rivals, total mass) and learns a better "keep this pair?" score q.
Trained on dec1's holdout pairs (ens1 probabilities + truth), 2-fold by record for an honest check,
then applied to ens1's test probabilities. Only Source-1-side features (the holdout has far fewer
competing Source 1 records per Source 2/3 record than the test, so Source-2/3-side features would shift).
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

KB = Path("D:/AmazonML/kaggle_build")
DEC = KB / "out_dec1/work/dec"
TEST_P = KB / "out_ens1/work/output/thresholds/test_pair_probabilities.parquet"
RAW = Path("D:/AmazonML/student_resource/dataset/test")
OUT = KB / "out_ctx1"
T0 = time.time()
LOG = []


def say(m):
    m = f"[{time.time() - T0:6.0f}s] {m}"
    print(m, flush=True)
    LOG.append(m)
    (OUT / "ctx_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def exclusive_sorted(s1_row, s23_row, p):
    order = np.lexsort((-p, s23_row))
    first = np.ones(len(order), dtype=bool)
    first[1:] = s23_row[order][1:] != s23_row[order][:-1]
    keep = order[first]
    return keep[np.lexsort((-p[keep], s1_row[keep]))]


def f05(n_pred, n_true, n_corr):
    precision = np.divide(n_corr, n_pred, out=np.zeros_like(n_corr), where=n_pred > 0)
    recall = np.divide(n_corr, n_true, out=np.zeros_like(n_corr), where=n_true > 0)
    bottom = 0.25 * precision + recall
    score = np.divide(1.25 * precision * recall, bottom, out=np.zeros_like(bottom), where=bottom > 0)
    return np.where((n_true == 0) & (n_pred == 0), 1.0, score)


def logit(x):
    x = np.clip(x.astype(np.float64), 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))


def features(s1_row, p):
    """s1_row sorted, p descending within each record (exclusive pairs)."""
    n = len(p)
    p = p.astype(np.float64)
    start = np.r_[True, s1_row[1:] != s1_row[:-1]]
    first = np.flatnonzero(start)
    group = np.cumsum(start) - 1
    rank = np.arange(n) - first[group]
    size = np.bincount(group)[group]
    lp = logit(p)
    p1 = p[first][group]
    has2 = size > 1
    p2 = np.where(has2, p[np.minimum(first[group] + 1, n - 1)], 0.0)
    prev = np.where(rank > 0, np.r_[1.0, p[:-1]], 1.0)
    last = rank == size - 1
    nxt = np.where(~last, np.r_[p[1:], 0.0], 0.0)
    cum = np.cumsum(p)
    before = np.r_[0.0, cum][first][group]
    above = cum - before - p
    total = np.bincount(group, weights=p)[group]
    f = {"p": p, "lp": lp, "rank": rank, "size": size, "p1": p1, "p2": p2,
         "gap_best": logit(p1) - lp, "gap_prev": logit(prev) - lp, "gap_next": lp - logit(nxt),
         "p_next": nxt, "p_prev": prev, "ratio_best": p / np.maximum(p1, 1e-9),
         "mass_above": above, "mass_total": total, "mass_below": total - above - p}
    for t in (0.9, 0.7, 0.5, 0.3, 0.1):
        f[f"n_gt{t}"] = np.bincount(group, weights=(p > t).astype(float))[group]
    return pd.DataFrame(f, dtype=np.float32), rank, group, first


def score(index, n_rows, keep, correct, n_true):
    n_pred = np.bincount(index[keep], minlength=n_rows).astype(float)
    n_corr = np.bincount(index[keep & correct], minlength=n_rows).astype(float)
    return f05(n_pred, n_true, n_corr)


def keep_rule(q, rank, t, t1=None):
    k = q >= t
    if t1 is not None:
        k |= (rank == 0) & (q >= t1)
    return k


PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=31, min_data_in_leaf=300,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0,
              verbose=-1, num_threads=4, seed=7)
ROUNDS = 500


def main():
    OUT.mkdir(exist_ok=True)
    pairs = pd.read_parquet(DEC / "holdout_pairs.parquet", columns=["s1_row", "s23_row", "probability", "label"])
    truth = pd.read_parquet(DEC / "holdout_truth.parquet")
    rows = pd.read_parquet(DEC / "holdout_rows.parquet").sort_values("s1_row").reset_index(drop=True)
    s1, s23, p = (pairs[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    o = exclusive_sorted(s1, s23, p)
    s1, s23, p = s1[o], s23[o], p[o]
    key = s1.astype(np.int64) * 10_000_000 + s23
    tkey = truth["s1_row"].to_numpy().astype(np.int64) * 10_000_000 + truth["s23_row"].to_numpy()
    correct = np.isin(key, tkey)
    rrows = rows["s1_row"].to_numpy()
    n_rows = len(rrows)
    n_true = np.bincount(np.searchsorted(rrows, truth["s1_row"].to_numpy()), minlength=n_rows).astype(float)
    index = np.searchsorted(rrows, s1)
    country = rows["country"].to_numpy()
    X, rank, group, first = features(s1, p)
    say(f"holdout: {n_rows:,} records, {len(p):,} exclusive pairs, {correct.sum():,} correct, {int(n_true.sum()):,} true links")

    fold_rec = (rrows.astype(np.uint64) * np.uint64(2654435761) % np.uint64(4294967296)) % np.uint64(2)
    fold = fold_rec.astype(int)[index]
    q = np.zeros(len(p))
    for k in (0, 1):
        tr = fold != k
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], correct[tr].astype(int)), ROUNDS)
        q[~tr] = m.predict(X[~tr])
    base = score(index, n_rows, keep_rule(p, rank, 0.70), correct, n_true)
    top1 = score(index, n_rows, keep_rule(p, rank, 0.74, 0.56), correct, n_true)
    say(f"baseline p T0.70: {base.mean():.5f}   dec1 TOP1(0.74,0.56): {top1.mean():.5f}")

    grid_t = np.round(np.arange(0.30, 0.92, 0.02), 2)
    grid_t1 = [None] + list(np.round(np.arange(0.30, 0.70, 0.04), 2))
    results = {}
    for t in grid_t:
        for t1 in grid_t1:
            if t1 is not None and t1 >= t:
                continue
            results[(t, t1)] = score(index, n_rows, keep_rule(q, rank, t, t1), correct, n_true)
    best = max(results, key=lambda c: results[c].mean())
    say(f"ctx q best {best}: {results[best].mean():.5f}  (gain vs T0.70 {results[best].mean() - base.mean():+.5f})")
    # honest: choose (t, t1) on one half of the records, measure on the other half
    honest = 0.0
    for k in (0, 1):
        pick = max(results, key=lambda c: results[c][fold_rec == k].mean())
        g = (results[pick] - base)[fold_rec != k].mean()
        honest += g * (fold_rec != k).mean()
        say(f"  pick on half {k}: {pick} -> gain on other half {g:+.5f}")
    say(f"2-fold honest gain vs T0.70: {honest:+.5f}   (dec1 TOP1 gain {top1.mean() - base.mean():+.5f})")
    for c in ("India", "US"):
        mk = country == c
        say(f"  {c}: T0.70 {base[mk].mean():.5f} -> ctx {results[best][mk].mean():.5f} ({(results[best] - base)[mk].mean():+.5f})")
    nt = n_true
    for name, mk in (("0", nt == 0), ("1", nt == 1), ("2-3", (nt >= 2) & (nt <= 3)), ("4+", nt >= 4)):
        say(f"  true matches {name}: {base[mk].mean():.4f} -> {results[best][mk].mean():.4f} ({mk.sum():,})")
    passes = honest > 0.0001 and all((results[best] - base)[country == c].mean() > 0 for c in ("India", "US"))
    say("PASSES" if passes else "does NOT pass (keep dec1 TOP1 / T0.70)")
    json.dump({"best": [best[0], best[1]], "honest_gain": honest, "passes": bool(passes),
               "holdout": results[best].mean(), "base": base.mean()}, open(OUT / "ctx_result.json", "w"))
    if "--test" not in sys.argv and not passes:
        return

    final = lgb.train(PARAMS, lgb.Dataset(X, correct.astype(int)), ROUNDS)
    final.save_model(str(OUT / "ctx_model.txt"))
    imp = sorted(zip(final.feature_importance("gain"), X.columns), reverse=True)[:8]
    say("importance: " + ", ".join(f"{n} {g:.0f}" for g, n in imp))
    del pairs, X
    test = pd.read_parquet(TEST_P)
    s1, s23, p = (test[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del test
    o = exclusive_sorted(s1, s23, p)
    s1, s23, p = s1[o], s23[o], p[o]
    del o
    Xt, rank, _, _ = features(s1, p)
    qt = final.predict(Xt, num_threads=4)
    del Xt
    say(f"test: {len(p):,} exclusive pairs scored")
    import pyarrow as pa
    import pyarrow.csv as pc

    def ids(name):
        t = pc.read_csv(RAW / name, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                        convert_options=pc.ConvertOptions(include_columns=["entity_id"],
                                                          column_types={"entity_id": pa.string()}))
        return t.column("entity_id").combine_chunks()
    s1_ids = ids("test_source1.tsv").to_numpy(zero_copy_only=False)
    s23_ids = pa.concat_arrays([ids("test_source2.tsv"), ids("test_source3.tsv")])

    def write(path, keep):
        a, b = s1[keep], s23[keep]
        sid = np.asarray(s23_ids.take(pa.array(b)).to_numpy(zero_copy_only=False), dtype=object)
        joined = pd.Series(sid).groupby(a).agg(",".join)
        col = joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": col}).to_csv(
            path, sep="\t", index=False, lineterminator="\n")
        say(f"wrote {path}: {keep.sum():,} pairs, {len(joined):,} records answered")

    write(OUT / "check_T0.70" / "matching_results.tsv", keep_rule(p, rank, 0.70))
    write(OUT / "output" / "matching_results.tsv", keep_rule(qt, rank, best[0], best[1]))
    pd.DataFrame({"s1_row": s1, "s23_row": s23, "probability": qt.astype(np.float32)}).to_parquet(
        OUT / "test_pair_probabilities_ctx.parquet", index=False)
    say("ALL DONE")


if __name__ == "__main__":
    main()

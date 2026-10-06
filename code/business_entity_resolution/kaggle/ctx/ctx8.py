"""
ctx8: the ctx3 re-scorer (ctx4.features, same LightGBM) RE-TRAINED on another pipeline's own holdout
(work/dec of ens3b etc.) instead of dec1's, then applied to that pipeline's test probabilities.
Honest check as ctx7: 2-fold by record, thresholds picked on one half, measured on the other; also
reports the absolute honest holdout F0.5 so pipelines can be compared on the same 66k records.
Usage: python ctx/ctx8.py <dec_dir> <tag> [--test <probs.parquet> <out_dir>] [--force]
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from ctx1 import exclusive_sorted, f05, keep_rule  # noqa: E402
import ctx4  # noqa: E402

KB = Path("D:/AmazonML/kaggle_build")
DEC = Path(sys.argv[1])
TAG = sys.argv[2]
OUT = KB / "out_ctx8" / TAG
LOG = []


def say(m):
    print(m, flush=True)
    LOG.append(m)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ctx8_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def main():
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
    say(f"{TAG}: {len(s1):,} exclusive holdout pairs on {n_rows:,} records from {DEC}")
    X, w = ctx4.features(s1, s23, p, "train")
    y = correct[w].astype(int)
    country = rows["country"].to_numpy()

    def grid(qq):
        o2 = np.lexsort((-qq, s1))
        r = np.empty(len(qq), dtype=np.int64)
        r[o2] = ctx4.rank_in_group(s1[o2])[0]
        res = {}
        for t in np.round(np.arange(0.50, 0.92, 0.02), 2):
            for t1 in [None] + list(np.round(np.arange(0.40, 0.78, 0.04), 2)):
                if t1 is not None and t1 >= t:
                    continue
                k = keep_rule(qq, r, t, t1)
                res[(t, t1)] = f05(np.bincount(index[k], minlength=n_rows).astype(float), n_true,
                                   np.bincount(index[k & correct], minlength=n_rows).astype(float))
        return res

    def honest(res):
        g = 0.0
        for k in (0, 1):
            pick = max(res, key=lambda c: res[c][fold_rec == k].mean())
            g += res[pick][fold_rec != k].sum()
        return g / n_rows

    rp = grid(p)
    q = p.copy()
    for k in (0, 1):
        tr, te = fold[w] != k, fold[w] == k
        m = lgb.train(ctx4.PARAMS, lgb.Dataset(X[tr], y[tr]), ctx4.ROUNDS)
        q[w[te]] = m.predict(X[te])
    rq = grid(q)
    pd.DataFrame({"s1_row": s1, "s23_row": s23, "p": p.astype(np.float32), "q": q.astype(np.float32),
                  "correct": correct}).to_parquet(OUT / "holdout_q.parquet", index=False)
    hp, hq = honest(rp), honest(rq)
    bp, bq = max(rp, key=lambda c: rp[c].mean()), max(rq, key=lambda c: rq[c].mean())
    say(f"honest holdout F0.5: raw p (best rule) {hp:.5f} -> re-scored {hq:.5f} ({hq - hp:+.5f})")
    say(f"best rules: raw {bp} {rp[bp].mean():.5f}; re-scored {bq} {rq[bq].mean():.5f}")
    for c in ("India", "US"):
        mk = country == c
        say(f"  {c}: raw {rp[bp][mk].mean():.5f} re-scored {rq[bq][mk].mean():.5f} ({mk.sum():,} records)")
    passes = hq - hp > 0.0002
    json.dump({"honest_raw": hp, "honest_ctx": hq, "best": [bq[0], bq[1]], "passes": bool(passes)},
              open(OUT / "ctx8_result.json", "w"))
    final = lgb.train(ctx4.PARAMS, lgb.Dataset(X, y), ctx4.ROUNDS)
    del X
    if "--test" in sys.argv and (passes or "--force" in sys.argv):
        i = sys.argv.index("--test")
        apply_test(final, Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), bq)


def apply_test(model, probs, out_dir, best):
    import pyarrow as pa
    import pyarrow.csv as pc
    d = pd.read_parquet(probs, columns=["s1_row", "s23_row", "probability"])
    s1, s23, p = (d[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del d
    o = exclusive_sorted(s1, s23, p)
    s1, s23, p = s1[o], s23[o], p[o].astype(np.float64)
    del o
    X, w = ctx4.features(s1, s23, p, "test")
    q = p.copy()
    q[w] = model.predict(X, num_threads=4)
    del X
    o = np.lexsort((-q, s1))
    s1, s23, q, p = s1[o], s23[o], q[o], p[o]
    rank, _, _ = ctx4.rank_in_group(s1)
    keep = keep_rule(q, rank, best[0], best[1])
    raw = Path("D:/AmazonML/student_resource/dataset/test")

    def ids(name):
        t = pc.read_csv(raw / name, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                        convert_options=pc.ConvertOptions(include_columns=["entity_id"],
                                                          column_types={"entity_id": pa.string()}))
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
                  "p_orig": p.astype(np.float32)}).to_parquet(out_dir / "test_pair_probabilities_ctx8.parquet",
                                                              index=False)
    say(f"test: wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs, {len(joined):,} records answered")


if __name__ == "__main__":
    main()

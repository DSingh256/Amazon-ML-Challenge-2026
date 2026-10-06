"""
judge_blend: add an LLM judge's score (judge1 one-token or judge2 chain-of-reasoning) on top of the
ctx3 re-scorer's q.
Holdout: q = ctx3 cross-fitted (2-fold by record, the same folds as ctx3/ctx7); for pairs the judge
scored, q' = logistic(logit q, score, logit q * score) fitted on the other fold; honest threshold
pick-on-half, measured on the records the judge scored (vs ctx3 q alone on the same records).
Test: q from v15 (ctx3 on wide30), q' where the judge scored the pair.
Usage: python ctx/judge_blend.py judge1|judge2 [--test <ctx3 test probs> <out_dir>] [--force]
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from ctx1 import exclusive_sorted, f05, keep_rule, logit  # noqa: E402
import ctx4  # noqa: E402

KB = Path("D:/AmazonML/kaggle_build")
DEC = KB / "out_dec1/work/dec"
KIND = sys.argv[1]
OUT = KB / f"out_blend_{KIND}"
LOG = []


def say(m):
    print(m, flush=True)
    LOG.append(m)
    OUT.mkdir(exist_ok=True)
    (OUT / "blend_log.txt").write_text("\n".join(LOG) + "\n", encoding="utf-8")


def judge_scores(part):
    """DataFrame s1_row, s23_row, score of the judge for the holdout ('h') or test ('t') pairs."""
    if KIND == "judge2":
        d = pd.read_parquet(KB / "out_judge2/work/judge/judge2_scores.parquet", columns=["part", "s1_row", "s23_row", "score"])
        return d[d["part"] == part].drop(columns="part").dropna(subset=["score"])
    if part == "t":
        return pd.read_parquet(KB / "out_judge1/work/judge/test_llm_scores.parquet", columns=["s1_row", "s23_row", "score"])
    raw = pd.read_parquet(KB / "out_judge1/work/judge/llm_scores_raw.parquet")
    raw = raw[raw["part"] == "h"]
    hold = pd.read_parquet(DEC / "holdout_pairs.parquet", columns=["s1_row", "s23_row", "probability"])
    band = hold[(hold["probability"] >= 0.03) & (hold["probability"] <= 0.98)].reset_index(drop=True)
    b = band.iloc[raw["idx"].to_numpy()]
    return pd.DataFrame({"s1_row": b["s1_row"].to_numpy(), "s23_row": b["s23_row"].to_numpy(),
                         "score": raw["score"].to_numpy()})


def attach(s1, s23, sc):
    key = s1.astype(np.int64) * 10_000_000 + s23
    skey = sc["s1_row"].to_numpy().astype(np.int64) * 10_000_000 + sc["s23_row"].to_numpy()
    return pd.Series(sc["score"].to_numpy(), index=skey).groupby(level=0).first().reindex(key).to_numpy()


def design(q, s):
    lq = logit(q)
    return np.column_stack([lq, s, lq * s / 10])


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
    X, w = ctx4.features(s1, s23, p, "train")
    y = correct[w].astype(int)
    q = p.copy()
    for k in (0, 1):
        tr, te = fold[w] != k, fold[w] == k
        m = lgb.train(ctx4.PARAMS, lgb.Dataset(X[tr], y[tr]), ctx4.ROUNDS)
        q[w[te]] = m.predict(X[te])
    del X
    s = attach(s1, s23, judge_scores("h"))
    has = ~np.isnan(s)
    scored_rec = np.zeros(n_rows, bool)
    scored_rec[index[has]] = True
    yy = correct.astype(int)
    say(f"{KIND}: holdout pairs with a judge score: {has.sum():,} on {scored_rec.sum():,} records "
        f"({yy[has].mean():.3f} matches)")
    q2 = q.copy()
    for k in (0, 1):
        tr, te = has & (fold != k), has & (fold == k)
        m = LogisticRegression(C=1.0, max_iter=1000).fit(design(q[tr], s[tr]), yy[tr])
        q2[te] = m.predict_proba(design(q[te], s[te]))[:, 1]

    def grid(qq):
        o2 = np.lexsort((-qq, s1))
        r = np.empty(len(qq), dtype=np.int64)
        r[o2] = ctx4.rank_in_group(s1[o2])[0]
        res = {}
        for t in np.round(np.arange(0.50, 0.92, 0.02), 2):
            for t1 in [None] + list(np.round(np.arange(0.40, 0.78, 0.04), 2)):
                if t1 is not None and t1 >= t:
                    continue
                kk = keep_rule(qq, r, t, t1)
                res[(t, t1)] = f05(np.bincount(index[kk], minlength=n_rows).astype(float), n_true,
                                   np.bincount(index[kk & correct], minlength=n_rows).astype(float))
        return res
    r3, rj = grid(q), grid(q2)
    sub = scored_rec

    def honest(res):
        g = 0.0
        for k in (0, 1):
            pick = max(res, key=lambda c: res[c][sub & (fold_rec == k)].mean())
            other = sub & (fold_rec != k)
            g += res[pick][other].sum()
        return g / sub.sum()
    h3, hj = honest(r3), honest(rj)
    gain_all = (hj - h3) * sub.mean()
    b3, bj = max(r3, key=lambda c: r3[c].mean()), max(rj, key=lambda c: rj[c].mean())
    say(f"on the {sub.sum():,} scored records (honest threshold pick-on-half): ctx3 {h3:.5f} -> ctx3+{KIND} {hj:.5f} "
        f"({hj - h3:+.5f}); = {gain_all:+.5f} on the whole holdout")
    say(f"best thresholds: ctx3 {b3} {r3[b3].mean():.5f}, blend {bj} {rj[bj].mean():.5f}")
    country = rows["country"].to_numpy()
    for c in ("India", "US"):
        mk = sub & (country == c)
        say(f"  {c}: ctx3 {r3[b3][mk].mean():.5f} blend {rj[bj][mk].mean():.5f} ({mk.sum():,} records)")
    passes = gain_all > 0.0002
    say(f"{KIND} blend PASSES" if passes else f"{KIND} blend does NOT pass (+0.0002 on the whole holdout)")
    blend = LogisticRegression(C=1.0, max_iter=1000).fit(design(q[has], s[has]), yy[has])
    say(f"blend coef {np.round(blend.coef_[0], 4).tolist()} intercept {blend.intercept_[0]:.3f}")
    json.dump({"honest_ctx3": h3, "honest_blend": hj, "gain_whole": gain_all, "best": [bj[0], bj[1]],
               "passes": bool(passes)}, open(OUT / "blend_result.json", "w"))
    if "--test" in sys.argv and (passes or "--force" in sys.argv):
        i = sys.argv.index("--test")
        apply_test(blend, Path(sys.argv[i + 1]), Path(sys.argv[i + 2]), bj)


def apply_test(blend, probs, out_dir, best):
    import pyarrow as pa
    import pyarrow.csv as pc
    d = pd.read_parquet(probs, columns=["s1_row", "s23_row", "probability"])
    s1, s23, q = (d[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
    del d
    q = q.astype(np.float64)
    s = attach(s1, s23, judge_scores("t"))
    has = ~np.isnan(s)
    q2 = q.copy()
    q2[has] = blend.predict_proba(design(q[has], s[has]))[:, 1]
    say(f"test: {has.sum():,} of {len(q):,} pairs have a judge score; crossing {best[0]}: "
        f"up {((q < best[0]) & (q2 >= best[0])).sum():,}, down {((q >= best[0]) & (q2 < best[0])).sum():,}")
    o = exclusive_sorted(s1, s23, q2)
    s1, s23, q2 = s1[o], s23[o], q2[o]
    o = np.lexsort((-q2, s1))
    s1, s23, q2 = s1[o], s23[o], q2[o]
    rank, _, _ = ctx4.rank_in_group(s1)
    keep = keep_rule(q2, rank, best[0], best[1])
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
    pd.DataFrame({"s1_row": s1, "s23_row": s23, "probability": q2.astype(np.float32)}).to_parquet(
        out_dir / f"test_pair_probabilities_{KIND}.parquet", index=False)
    say(f"test: wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs, {len(joined):,} records answered")


if __name__ == "__main__":
    main()

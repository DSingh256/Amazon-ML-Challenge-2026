"""
Diagnostic: WHY are true pairs missing from the candidates? (one country, part of Source 1)

Runs the normal blocking for one country of the train data, but only for a part of the
Source 1 records, and keeps every search result. Then it answers:
  - how much recall is lost by keeping only the best 30 candidates per record
    (the same hits trimmed to 30 / 45 / 60 / no limit)?
  - how much by the searches themselves (true pairs that are in NO search result)?
  - what do the never-found pairs look like (text similarity, examples)?
  - does a wider forward search (more pieces per query, larger k) find them?

    python src/diag_blocking.py --data-dir <folder with train/> --work-dir <folder> --country India
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from sklearn.feature_extraction.text import TfidfVectorizer

from blocking import combine_and_trim, field_patterns, run_retriever, search
from config import BLOCKING, Retriever
from evaluate import blocking_report
from io_utils import id_bucket, log, read_ground_truth_pairs, truth_as_rows
from run_blocking import load_clean

PAIR = ["s1_row", "s23_row"]


def pair_key(table):
    return table["s1_row"].to_numpy(np.int64) * (1 << 32) + table["s23_row"].to_numpy(np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--country", default="India")
    parser.add_argument("--fraction", type=float, default=0.14, help="part of Source 1 used as queries")
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--variants-only", action="store_true", help="skip the full report, only try the search variants")
    args = parser.parse_args(argv)
    split_dir = Path(args.work_dir) / "train"
    split_dir.mkdir(parents=True, exist_ok=True)
    report = []

    def say(text=""):
        report.append(text)
        print(text, flush=True)

    s1, s23 = load_clean(args.data_dir, "train", split_dir, args.n_jobs)
    truth = truth_as_rows(read_ground_truth_pairs(args.data_dir), s1, s23)
    s1_group = s1[s1["country"] == args.country]
    s23_group = s23[s23["country"] == args.country]
    is_query = id_bucket(s1_group["entity_id"]) < args.fraction * 1000
    query_rows = s1_group.index.to_numpy()[is_query]
    log(f"{args.country}: {is_query.sum():,} query records of {len(s1_group):,}")
    truth_q = truth[truth["s1_row"].isin(query_rows)].reset_index(drop=True)
    say(f"country {args.country}: {len(query_rows):,} query records, {len(truth_q):,} true pairs")

    if not args.variants_only:
        # the normal searches (forward 2, forward 3, reverse), all hits kept
        retriever = BLOCKING.retrievers[0]
        hits = run_retriever(s1_group, s23_group, retriever, BLOCKING, is_query.to_numpy() if hasattr(is_query, "to_numpy") else is_query,
                             False, split_dir / f"search_{args.country}_diag")
        hits.to_parquet(split_dir / "diag_hits.parquet")
        log(f"{len(hits):,} raw hits")

        true_key = pair_key(truth_q)
        hit_key = pair_key(hits)
        say()
        say("recall BEFORE any trimming, by search:")
        for method in sorted(hits["method"].unique()):
            say(f"   {method:<16} {np.isin(true_key, hit_key[(hits['method'] == method).to_numpy()]).mean():.4f}")
        found_any = np.isin(true_key, hit_key)
        say(f"   all searches together (no cap)  {found_any.mean():.4f}")
        per_rank = hits[["s1_row", "s23_row", "rank", "method"]]
        for method in sorted(hits["method"].unique()):
            sub = per_rank[per_rank["method"] == method]
            sub_key = pair_key(sub)
            ranks = pd.Series(sub["rank"].to_numpy()).groupby(sub_key).min()
            found = pd.Series(ranks.reindex(true_key).to_numpy())
            say(f"   {method}: rank of true pairs (0 = best)  " +
                "  ".join(f"<={k}: {(found <= k).mean():.4f}" for k in (0, 1, 2, 4, 9, 14)))

        # the same hits kept at different caps
        for cap in (30, 45, 60, 100000):
            candidates, _ = combine_and_trim(hits, cap)
            say()
            say(f"=== cap {cap} candidates per record: {len(candidates) / len(query_rows):.1f} per record ===")
            say(blocking_report(s1, candidates, truth, query_rows))

        # what do the never-found pairs look like?
        missed = truth_q[~found_any].reset_index(drop=True)
        say()
        say(f"never found by any search: {len(missed):,} of {len(truth_q):,} true pairs "
            f"({len(missed) / len(truth_q):.4f})")
        if len(missed):
            sample = missed.sample(min(len(missed), 20000), random_state=0)
            text1 = s1["name_addr_text"].to_numpy()[sample["s1_row"].to_numpy()]
            text23 = s23["name_addr_text"].to_numpy()[sample["s23_row"].to_numpy()]
            similarity = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(text1, text23)])
            found_sample = truth_q[found_any].sample(min(found_any.sum(), 20000), random_state=0)
            sim_found = np.array([fuzz.token_set_ratio(a, b) for a, b in zip(
                s1["name_addr_text"].to_numpy()[found_sample["s1_row"].to_numpy()],
                s23["name_addr_text"].to_numpy()[found_sample["s23_row"].to_numpy()])])
            say("text similarity (token_set_ratio, 0-100) quantiles 10/25/50/75/90:")
            say("   missed pairs : " + str(np.percentile(similarity, [10, 25, 50, 75, 90]).round(0)))
            say("   found pairs  : " + str(np.percentile(sim_found, [10, 25, 50, 75, 90]).round(0)))
            say("text length (characters) quantiles 10/25/50/75/90 of the Source 1 text:")
            say("   missed pairs : " + str(np.percentile([len(t) for t in text1], [10, 25, 50, 75, 90])))
            say("   found pairs  : " + str(np.percentile(
                [len(t) for t in s1["name_addr_text"].to_numpy()[found_sample["s1_row"].to_numpy()]], [10, 25, 50, 75, 90])))
            say()
            say("examples of never-found pairs (Source 1 text | Source 2/3 text):")
            for _, row in missed.sample(min(len(missed), 40), random_state=1).iterrows():
                say(f"   {s1['name_addr_text'].iloc[row['s1_row']]!r}  |  {s23['name_addr_text'].iloc[row['s23_row']]!r}")

    # a wider forward search on a smaller part of the queries
    small = id_bucket(s1_group["entity_id"]) < args.fraction * 1000 * 0.3
    small_rows = s1_group.index.to_numpy()[small]
    truth_small = truth[truth["s1_row"].isin(small_rows)]
    small_key = pair_key(truth_small)
    say()
    say(f"wider forward search on {len(small_rows):,} query records ({len(truth_small):,} true pairs):")
    # other ways of searching (name only, address only ...): recall and work, on the same small part
    say()
    say(f"other retrievers on the {len(small_rows):,} query records "
        f"(reverse: only the true Source 2/3 records are used as queries; that gives exactly the same "
        f"hits for the true pairs as the full reverse search)")
    small_np = small.to_numpy() if hasattr(small, "to_numpy") else small
    s1_rows_all = s1_group.index.to_numpy()
    s23_rows_all = s23_group.index.to_numpy()
    query_positions = np.flatnonzero(small_np)
    true_s23_positions = np.unique(pd.Index(s23_rows_all).get_indexer(truth_small["s23_row"].to_numpy()))
    found_sets = {}

    def variant(name, column, config, k_forward, k_reverse, fields=(), coverage=0.0, forward=True):
        vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), sublinear_tf=True, dtype=np.float32)
        vectorizer.fit(pd.concat([s1_group[column], s23_group[column]]))
        m1 = vectorizer.transform(s1_group[column])
        m23 = vectorizer.transform(s23_group[column])
        p1 = field_patterns(vectorizer, s1_group, fields)
        p23 = field_patterns(vectorizer, s23_group, fields)
        keys = {}
        for source in (2, 3):
            doc_positions = np.flatnonzero(s23_group["source"].to_numpy() == source)
            found = search(m1[query_positions], m23[doc_positions], k_forward, config,
                           patterns=[p[query_positions] for p in p1])
            keys[f"fwd{source}"] = pair_key(pd.DataFrame({
                "s1_row": s1_rows_all[query_positions[found["query"].astype(int)]],
                "s23_row": s23_rows_all[doc_positions[found["doc"].astype(int)]]}))
        found = search(m23[true_s23_positions], m1, k_reverse, config,
                       patterns=[p[true_s23_positions] for p in p23], coverage=coverage)
        keys["rev"] = pair_key(pd.DataFrame({
            "s1_row": s1_rows_all[found["doc"].astype(int)],
            "s23_row": s23_rows_all[true_s23_positions[found["query"].astype(int)]]}))
        all_keys = np.unique(np.concatenate(list(keys.values())))
        found_sets[name] = all_keys
        say(f"   {name}: forward {np.isin(small_key, np.concatenate([keys['fwd2'], keys['fwd3']])).mean():.4f}  "
            f"reverse {np.isin(small_key, keys['rev']).mean():.4f}  both {np.isin(small_key, all_keys).mean():.4f}")

    W = replace(BLOCKING, max_query_ngrams=60, min_query_ngrams=30, max_ngram_share=0.005)
    F = ("name_text", "addr_text")
    NA = "name_addr_text"
    variant("current k15/3", NA, BLOCKING, 15, 3)
    variant("current, rev k10", NA, BLOCKING, 15, 10)
    variant("current, rev k10 cov 0.7", NA, BLOCKING, 15, 10, coverage=0.7)
    variant("current, rev k10 cov 1.0", NA, BLOCKING, 15, 10, coverage=1.0)
    variant("fields 14/16 rev k10 cov 0.7", NA, BLOCKING, 15, 10, F, 0.7)
    variant("fields 20/20 (40) rev k10 cov 0.7", NA, replace(BLOCKING, max_query_ngrams=40, field_quota=(20, 20)), 15, 10, F, 0.7)
    variant("wide rev k10 cov 0.7", NA, W, 15, 10, coverage=0.7)
    variant("wide fields 30/30 (60) rev k10 cov 0.7", NA,
            replace(W, field_quota=(30, 30)), 15, 10, F, 0.7)
    variant("wide fields 30/30 fwd k30 rev k10 cov 0.7", NA,
            replace(W, field_quota=(30, 30)), 30, 10, F, 0.7)
    base = found_sets["current k15/3"]
    say("union with the current search:")
    for name, keys in found_sets.items():
        say(f"   current + {name}: {np.isin(small_key, np.union1d(base, keys)).mean():.4f}")

    (split_dir / "diag_report.txt").write_text("\n".join(report), encoding="utf-8")
    log("diagnostic finished")


if __name__ == "__main__":
    main()

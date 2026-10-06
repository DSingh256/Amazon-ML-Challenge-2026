"""
Run S0 (load) -> S1 (clean) -> S2 (blocking) for one split.

Train split : also measures how many true matches the candidates contain.
Test split  : also writes candidate_pairs.tsv in the submission format.

Examples:
    # laptop, on the small sample
    python src/run_blocking.py --data-dir D:/AmazonML/work/sample_data --split train \
                               --work-dir D:/AmazonML/work/sample_run

    # only print how much work the search would be (to estimate the time)
    python src/run_blocking.py --data-dir ... --split test --work-dir ... --dry-run
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from blocking import build_candidates
from config import BLOCKING
from evaluate import blocking_report
from io_utils import (id_bucket, log, read_ground_truth_pairs, read_source1, read_source23,
                      truth_as_rows, write_id_lists)
from normalize import normalize_table


def load_clean(data_dir, split, split_dir, n_jobs):
    """Load and clean Source 1 and Source 2/3. Cleaned tables are saved,
    so a second run skips this slow step."""
    s1_file = split_dir / "s1_clean.parquet"
    s23_file = split_dir / "s23_clean.parquet"
    if s1_file.exists() and s23_file.exists():
        log("loading cleaned tables from cache")
        return pd.read_parquet(s1_file), pd.read_parquet(s23_file)

    s1 = read_source1(data_dir, split)
    s23 = read_source23(data_dir, split)
    log(f"loaded {len(s1):,} Source 1 and {len(s23):,} Source 2/3 records")

    s1 = normalize_table(s1, n_jobs)
    s23 = normalize_table(s23, n_jobs)
    log("cleaning done")

    split_dir.mkdir(parents=True, exist_ok=True)
    s1.to_parquet(s1_file, index=False)
    s23.to_parquet(s23_file, index=False)
    return s1, s23


def pick_query_rows(s1, fraction):
    """Choose which Source 1 records get candidates. With fraction < 1 we take a
    fixed random part (same ids every run, because it depends only on the id)."""
    if fraction >= 1:
        return np.ones(len(s1), dtype=bool)
    return id_bucket(s1["entity_id"]) < fraction * 1000


def run_blocking(data_dir, split, split_dir, n_jobs=1, query_fraction=1.0, dry_run=False, countries=None):
    """Clean + blocking for one split; saves candidates.parquet (and for test
    candidate_pairs.tsv, for train blocking_report.txt) in split_dir.
    With `countries` only those countries are searched and their candidates_part_*.parquet
    saved (no candidates.parquet): a later run with all parts as input combines them."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    # S0 + S1
    s1, s23 = load_clean(data_dir, split, split_dir, n_jobs)

    # S2
    query_mask = pick_query_rows(s1, query_fraction)
    candidates = build_candidates(s1, s23, BLOCKING, query_mask, dry_run=dry_run,
                                  checkpoint_dir=split_dir, countries=countries)
    if countries is not None:
        log(f"blocking of {', '.join(countries)} done ({len(candidates):,} candidate pairs)")
        return
    if dry_run:
        log("dry run: nothing saved")
        return
    candidates.to_parquet(split_dir / "candidates.parquet", index=False)
    log(f"saved {len(candidates):,} candidate pairs")

    if split == "test":
        write_id_lists(split_dir / "candidate_pairs.tsv", s1["entity_id"], candidates,
                       s23["entity_id"], header="candidate_entity_ids")
        log(f"wrote {split_dir / 'candidate_pairs.tsv'}")
    else:
        truth = truth_as_rows(read_ground_truth_pairs(data_dir), s1, s23)
        report = blocking_report(s1, candidates, truth, np.flatnonzero(query_mask), s23)
        (split_dir / "blocking_report.txt").write_text(report)
        print("\n" + report + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder that contains train/ and test/")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--work-dir", required=True, help="results go to <work-dir>/<split>")
    parser.add_argument("--query-fraction", type=float, default=1.0,
                        help="part of Source 1 to make candidates for (quick recall checks)")
    parser.add_argument("--n-jobs", type=int, default=1, help="CPU cores for cleaning")
    parser.add_argument("--dry-run", action="store_true", help="only print the search work")
    args = parser.parse_args()
    run_blocking(args.data_dir, args.split, Path(args.work_dir) / args.split,
                 args.n_jobs, args.query_fraction, args.dry_run)
    log("done")


if __name__ == "__main__":
    main()

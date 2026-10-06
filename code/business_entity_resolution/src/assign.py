"""
S6 + S7 - From match probabilities to the final answer file.

S6  Exclusivity : a Source 2/3 record belongs to at most ONE Source 1 record
                  (true for all training data). When several Source 1 records
                  want the same record, only the one with the highest
                  probability keeps it.

S7  Decision    : for every Source 1 record, decide how many of its candidates
                  to output.  Two ways:
        "threshold" : keep every pair with probability >= t (t tuned on holdout)
        "expected"  : pick, per record, the list with the best EXPECTED F0.5

    Expected F0.5 in simple words. Sort one record's candidates by probability
    p1 >= p2 >= ... and think of "answer with the top k".
        expected number of true matches   T  = p1 + p2 + ... (all candidates)
        expected correct in our answer   TP  = p1 + ... + pk
        F0.5 = 1.25 * TP / (0.25 * T + k)
        empty answer (k = 0) is right only if the record has no match at all,
        probability P0 = (1-p1)(1-p2)...   -> expected score P0
        non-empty answer scores 0 when the record has no match, so
        expected score = (1 - P0) * 1.25 * TP / (0.25 * T + k)
    We take the k with the highest expected score. This says "answer nothing"
    for records where every candidate is doubtful (singletons are 5.6% of Source 1)
    and answers a long list for records where many candidates are sure.

All pair tables use row numbers of the cleaned tables: columns s1_row, s23_row.

Usage (make the final files for the test split):
    python src/assign.py --work-dir D:/AmazonML/work/run
"""
import argparse
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from io_utils import log, write_id_lists

BETA_SQUARED = 0.25          # F0.5


# ---------------------------------------------------------------------------
# S6: exclusivity
# ---------------------------------------------------------------------------
def apply_exclusivity(pairs):
    """pairs has columns s1_row, s23_row, probability.
    Keeps only the highest-probability row for every s23_row."""
    best_first = pairs.sort_values("probability", ascending=False, kind="stable")
    return best_first.drop_duplicates("s23_row", keep="first")


# ---------------------------------------------------------------------------
# S7: decisions
# ---------------------------------------------------------------------------
def pick_by_threshold(pairs, threshold):
    """Keep pairs with probability >= threshold."""
    return pairs[pairs["probability"] >= threshold][["s1_row", "s23_row"]]


def pick_by_expected_f05(pairs):
    """For each Source 1 record keep the top-k candidates with the best expected F0.5."""
    pairs = pairs.sort_values(["s1_row", "probability"], ascending=[True, False]).reset_index(drop=True)
    p = pairs["probability"].clip(0, 1 - 1e-6)
    by_record = pairs["s1_row"]

    k = p.groupby(by_record).cumcount() + 1                    # 1, 2, 3 ... inside each record
    expected_correct = p.groupby(by_record).cumsum()           # p1 + ... + pk
    expected_true = p.groupby(by_record).transform("sum")      # p1 + ... + all
    p_no_match = np.exp(np.log1p(-p).groupby(by_record).transform("sum"))   # (1-p1)(1-p2)...

    score_of_k = ((1 - p_no_match) * (1 + BETA_SQUARED) * expected_correct
                  / (BETA_SQUARED * expected_true + k))
    best_score = score_of_k.groupby(by_record).transform("max")
    best_k = k.where(score_of_k == best_score).groupby(by_record).transform("min")

    answer_is_empty = p_no_match >= best_score                 # empty list is at least as good
    keep = (k <= best_k) & ~answer_is_empty
    return pairs.loc[keep, ["s1_row", "s23_row"]]


# The expected-F0.5 decoder trusts the probabilities; on a country the model has not seen they
# are less right, and the unseen-country check showed the threshold decoder (threshold tuned on
# the holdout) doing better there (US learned from India: 0.9710 vs 0.9654, India from US:
# 0.9103 vs 0.9010) while the two are equal on the holdout. So "expected" must win clearly.
DECODER_MARGIN = 0.001


def choose_decoder(expected_f05, threshold_f05):
    return "expected" if expected_f05 > threshold_f05 + DECODER_MARGIN else "threshold"


def decide(pairs, method, threshold=None, exclusivity=True):
    """Full decision: exclusivity (S6) then threshold or expected-F0.5 (S7)."""
    if exclusivity:
        pairs = apply_exclusivity(pairs)
    if method == "threshold":
        return pick_by_threshold(pairs, threshold)
    if method == "expected":
        return pick_by_expected_f05(pairs)
    raise ValueError(f"unknown method {method!r}")


# ---------------------------------------------------------------------------
# Final files for one split
# ---------------------------------------------------------------------------
def score_pairs(features_path, model_dir, batch_size=1_000_000):
    """Probability of every pair in features.parquet, read in batches (the full
    test set is too big to load at once). Returns (pairs table, meta).
    With a stage-2 model (meta "stage1_models", see stage2.py): first the mean of the
    stage-1 models for all pairs, then the collective features, then stage 2."""
    model_dir = Path(model_dir)
    meta = json.loads((model_dir / "meta.json").read_text())
    model = lgb.Booster(model_file=str(model_dir / "lgbm.txt"))
    stage1 = [lgb.Booster(model_file=str(model_dir / name)) for name in meta.get("stage1_models", [])]
    first_columns = meta["stage1_features"] if stage1 else meta["features"]
    file = pq.ParquetFile(features_path)

    def batches(columns):
        for batch in file.iter_batches(batch_size=batch_size, columns=["s1_row", "s23_row"] + columns):
            yield batch.to_pandas()

    parts = []
    for batch in batches(first_columns):
        models = stage1 or [model]
        p = sum(m.predict(batch[first_columns]) for m in models) / len(models)
        parts.append(pd.DataFrame({"s1_row": batch["s1_row"].to_numpy(),
                                   "s23_row": batch["s23_row"].to_numpy(),
                                   "probability": p.astype(np.float32)}))
    pairs = pd.concat(parts, ignore_index=True)
    del parts
    log(f"scored {len(pairs):,} pairs" + (" (stage 1)" if stage1 else ""))
    if not stage1 or not meta.get("use_stage2", True):
        return pairs, meta

    from stage2 import STAGE2_COLUMNS, collective_features
    s23 = pd.read_parquet(Path(features_path).parent / "s23_clean.parquet", columns=["name_core", "addr_clean"])
    rival_tables = None
    try:
        from rival import load_tables
        rival_tables = load_tables(Path(features_path).parent)
    except FileNotFoundError as error:
        log(f"rival features skipped (cleaned tables not found: {error})")
    extra = collective_features([(pairs["s1_row"].to_numpy(), pairs["s23_row"].to_numpy(),
                                  pairs["probability"].to_numpy())],
                                s23["name_core"].to_numpy(), s23["addr_clean"].to_numpy(), rival_tables)[0]
    del s23
    log("collective features ready")
    if meta.get("cross_encoder"):
        from cross_encoder import SCORE_FILE, ce_features
        scores = pd.read_parquet(Path(features_path).parent / SCORE_FILE)
        extra = pd.concat([extra, ce_features(pairs["s1_row"], pairs["s23_row"], scores)], axis=1)
        log(f"cross-encoder scores of {len(scores):,} pairs added")
        del scores
    probability = np.empty(len(pairs), dtype=np.float32)
    start = 0
    for batch in batches(first_columns):
        stop = start + len(batch)
        batch = pd.concat([batch.reset_index(drop=True), extra.iloc[start:stop].reset_index(drop=True)], axis=1)
        probability[start:stop] = model.predict(batch[meta["features"]])
        start = stop
    pairs["probability"] = probability
    log(f"scored {len(pairs):,} pairs (stage 2)")
    return pairs, meta


def write_outputs(split_dir, model_dir, output_dir, method="auto"):
    """Make output/matching_results.tsv (+ copy candidate_pairs.tsv) for one split."""
    split_dir, output_dir = Path(split_dir), Path(output_dir)
    pairs, meta = score_pairs(split_dir / "features.parquet", model_dir)
    if method == "auto":
        holdout = json.loads((Path(model_dir) / "metrics.json").read_text())["holdout"]
        method = choose_decoder(holdout["macro_f05_expected"], holdout["macro_f05_threshold"])
    chosen = decide(pairs, method, threshold=meta["threshold"])
    log(f"decoder = {method}; {len(chosen):,} pairs in the final answer")

    # One row per Source 1 record, in the original file order (= row order).
    s1_ids = pd.read_parquet(split_dir / "s1_clean.parquet", columns=["entity_id"])["entity_id"]
    s23_ids = pd.read_parquet(split_dir / "s23_clean.parquet", columns=["entity_id"])["entity_id"]
    write_id_lists(output_dir / "matching_results.tsv", s1_ids, chosen, s23_ids, header="matched_entity_ids")
    shutil.copy(split_dir / "candidate_pairs.tsv", output_dir / "candidate_pairs.tsv")
    log(f"wrote {output_dir / 'matching_results.tsv'} and candidate_pairs.tsv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True, help="has test/ (features, candidates) and models/")
    parser.add_argument("--model-dir", default=None, help="default: <work-dir>/models")
    parser.add_argument("--method", choices=["auto", "threshold", "expected"], default="auto",
                        help="auto = see choose_decoder (holdout scores in metrics.json)")
    parser.add_argument("--output-dir", default=None, help="default: <work-dir>/output")
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    write_outputs(work_dir / "test", args.model_dir or work_dir / "models",
                  args.output_dir or work_dir / "output", args.method)


if __name__ == "__main__":
    main()

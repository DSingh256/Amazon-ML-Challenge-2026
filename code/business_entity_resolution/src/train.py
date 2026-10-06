"""
S4 - Train the pair model (LightGBM), and measure it (S8).

Learns from features.parquet: "is this candidate pair a real match?" (label 0/1).

We hold out 10% of the training Source 1 records (never used for learning) and measure:
  - pair AUC      : how well the probabilities rank matches above non-matches
  - macro F0.5    : the official score, for each way of deciding (see assign.py)
  - a breakdown by country / number of matches / missing address
  - the worst mistakes, saved to errors.tsv (read them to find new features)
One line per run is added to results.md.

Optional --transfer-check: train without one country, test on it. This shows
how well the model will do on an unseen country (France in the test data).

Usage:
    python src/train.py --work-dir D:/AmazonML/work/sample_run --data-dir D:/AmazonML/work/sample_data
"""
import argparse
import gc
import json
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from assign import DECODER_MARGIN, choose_decoder, decide
from config import HOLDOUT_PERCENT, SEED
from cross_encoder import CE_COLUMNS, SCORE_FILE, ce_features
from evaluate import error_examples, macro_f05, score_breakdown
from features import feature_columns, training_records
from io_utils import id_bucket, log, read_ground_truth_pairs, truth_as_rows
from stage2 import STAGE2_COLUMNS, collective_features

LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "seed": SEED,
    "verbose": -1,
    "metric": ["binary_logloss", "auc"],
}
# The full-data run (train2) was still improving a little at round 2000; early stopping ends it sooner when done.
MAX_ROUNDS = 3000
EARLY_STOP = 50
# Stage 1 (only feeds stage 2): faster settings. On the full data the log-loss hardly moved
# after ~1000 rounds at learning rate 0.05.
STAGE1_PARAMS = {**LGBM_PARAMS, "learning_rate": 0.1}
STAGE1_MAX_ROUNDS = 800
PREDICT_BATCH = 2_000_000
# Columns read from the cleaned tables (the full tables with every text column need ~10 GB).
S1_COLUMNS = ["entity_id", "business_name", "business_address", "country", "addr_clean"]
S23_COLUMNS = ["entity_id", "business_name", "business_address", "name_core", "addr_clean"]
THRESHOLDS = np.round(np.arange(0.30, 0.91, 0.05), 2)


def predict(model, frame, columns):
    """Probabilities in batches (less memory than one big predict)."""
    out = np.empty(len(frame), dtype=np.float32)
    for start in range(0, len(frame), PREDICT_BATCH):
        part = frame.iloc[start:start + PREDICT_BATCH]
        out[start:start + len(part)] = model.predict(part[columns], num_iteration=model.best_iteration)
    return out


def train_model(train, valid, columns, params=LGBM_PARAMS, max_rounds=MAX_ROUNDS):
    """Fit LightGBM with early stopping on `valid`.
    Returns (booster, curve): curve has one row per boosting round ("epoch") with the
    train and validation log-loss and AUC."""
    train_set = lgb.Dataset(train[columns], train["label"])
    valid_set = lgb.Dataset(valid[columns], valid["label"], reference=train_set)
    history = {}
    model = lgb.train(
        params, train_set, num_boost_round=max_rounds,
        valid_sets=[train_set, valid_set], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(EARLY_STOP, first_metric_only=True, verbose=False),
                   lgb.log_evaluation(50), lgb.record_evaluation(history)],
    )
    curve = pd.DataFrame({f"{name}_{metric}": values
                          for name, metrics in history.items() for metric, values in metrics.items()})
    curve.insert(0, "round", np.arange(1, len(curve) + 1))
    return model, curve


def save_curve(curve, best_round, out_dir):
    """training_curve.csv + training_curve.png (loss and AUC per round, best round marked)."""
    curve.to_csv(out_dir / "training_curve.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("train", "valid"):
        left.plot(curve["round"], curve[f"{name}_binary_logloss"], label=name)
        right.plot(curve["round"], curve[f"{name}_auc"], label=name)
    for axis, title in ((left, "log-loss (lower is better)"), (right, "AUC (higher is better)")):
        axis.axvline(best_round, color="grey", linestyle="--", label=f"best round {best_round}")
        axis.set_xlabel("boosting round")
        axis.set_title(title)
        axis.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "training_curve.png", dpi=120)
    plt.close(fig)


def best_threshold(scored, s1_rows, truth):
    """Try many thresholds; return (best threshold, its macro F0.5, all results)."""
    results = {float(t): macro_f05(s1_rows, decide(scored, "threshold", t), truth)
               for t in THRESHOLDS}
    best = max(results, key=results.get)
    return best, results[best], results


def best_of(result):
    return max(result["threshold_f05"], result["expected_f05"])


def cross_fit_stage1(learning, stopping, holdout, columns, fold):
    """Stage 1 twice, on two halves of the records (fold = 0/1 per Source 1 row).
    Every learn / early-stop pair gets the probability of the model that did NOT learn
    from its record; holdout pairs get the mean of both (as the test pairs will).
    Returns (models, [p_learning, p_stopping, p_holdout])."""
    fold_learn = fold[learning["s1_row"].to_numpy()]
    fold_stop = fold[stopping["s1_row"].to_numpy()]
    p_learn = np.zeros(len(learning), dtype=np.float32)
    p_stop = np.zeros(len(stopping), dtype=np.float32)
    p_hold = np.zeros(len(holdout), dtype=np.float32)
    models = []
    for k in (0, 1):
        model, curve = train_model(learning[fold_learn == k], stopping[fold_stop == k], columns,
                                   STAGE1_PARAMS, STAGE1_MAX_ROUNDS)
        log(f"stage 1, half {k}: best round {model.best_iteration} of {len(curve)}, valid log-loss "
            f"{curve['valid_binary_logloss'].iloc[model.best_iteration - 1]:.5f}")
        p_learn[fold_learn != k] = predict(model, learning[fold_learn != k], columns)
        p_stop[fold_stop != k] = predict(model, stopping[fold_stop != k], columns)
        p_hold += predict(model, holdout, columns) / 2
        models.append(model)
    return models, [p_learn, p_stop, p_hold]


def evaluate(model, columns, part, s1_rows, truth, title, probability=None):
    """Score one held-out part. Prints AUC and the macro F0.5 of each way of deciding:
      threshold (no exclusivity) | threshold + exclusivity (S6) | expected-F0.5 + exclusivity (S7)
    s1_rows: ALL Source 1 rows being checked (also those without any candidate).
    Returns a dict with the numbers and the scored pairs."""
    if probability is None:
        probability = predict(model, part, columns)
    part = part.assign(probability=probability)
    auc = roc_auc_score(part["label"], part["probability"])
    threshold, with_exclusivity, results = best_threshold(part, s1_rows, truth)
    without_exclusivity = macro_f05(s1_rows, decide(part, "threshold", threshold, exclusivity=False), truth)
    expected = macro_f05(s1_rows, decide(part, "expected"), truth)
    log(f"{title}: AUC {auc:.4f}")
    log(f"    threshold {threshold:.2f}, no exclusivity : {without_exclusivity:.4f}")
    log(f"    threshold {threshold:.2f}, exclusivity    : {with_exclusivity:.4f}")
    log(f"    expected-F0.5, exclusivity   : {expected:.4f}")
    method = "expected" if expected > with_exclusivity else "threshold"
    chosen = decide(part, method, threshold)
    true_pairs = truth[truth["s1_row"].isin(s1_rows)]
    hits = len(chosen.merge(true_pairs, on=["s1_row", "s23_row"]))
    precision = hits / max(len(chosen), 1)
    recall = hits / max(len(true_pairs), 1)
    log(f"    pairs ({method}): precision {precision:.4f}, recall {recall:.4f}")
    return {"threshold": threshold, "threshold_f05": with_exclusivity, "expected_f05": expected,
            "no_exclusivity_f05": without_exclusivity, "auc": float(auc),
            "precision": precision, "recall": recall,
            "all_thresholds": results, "scored": part}


def transfer_check(features, s1, s1_rows, truth, columns, n_trees):
    """Train without one country, test on it (like France in the test data).
    Uses the same number of trees as the main model (no peeking at the test country).
    Returns {country: best macro F0.5}."""
    country = s1["country"].to_numpy()
    pair_country = country[features["s1_row"].to_numpy()]
    scores = {}
    for name in sorted(np.unique(pair_country)):
        train_part = features[pair_country != name]
        test_part = features[pair_country == name]
        model = lgb.train(LGBM_PARAMS, lgb.Dataset(train_part[columns], train_part["label"]),
                          num_boost_round=n_trees)
        rows = s1_rows[country[s1_rows] == name]
        result = evaluate(model, columns, test_part, rows, truth, f"train without {name}, test on {name}")
        scores[name] = max(result["threshold_f05"], result["expected_f05"])
    return scores


def append_results(path, note, holdout_score, transfer_scores):
    """One line per experiment in results.md (the S8 log)."""
    path = Path(path)
    if not path.exists():
        path.write_text("| when | change | holdout F0.5 | unseen-country F0.5 | public LB |\n"
                        "|---|---|---|---|---|\n")
    transfer = ", ".join(f"{k} {v:.4f}" for k, v in transfer_scores.items()) or "-"
    with path.open("a") as f:
        f.write(f"| {datetime.now():%Y-%m-%d %H:%M} | {note} | {holdout_score:.4f} | {transfer} | |\n")


def run_training(work_dir, data_dir, transfer=False, note="", stage2=True):
    """Train on <work_dir>/train, save the model in <work_dir>/models."""
    work_dir = Path(work_dir)
    split_dir = work_dir / "train"
    features = pd.read_parquet(split_dir / "features.parquet")
    s1 = pd.read_parquet(split_dir / "s1_clean.parquet", columns=S1_COLUMNS)
    s23 = pd.read_parquet(split_dir / "s23_clean.parquet", columns=S23_COLUMNS)
    truth = truth_as_rows(read_ground_truth_pairs(data_dir), s1, s23)
    columns = feature_columns(features.columns)
    log(f"{len(features):,} pairs, {len(columns)} features")

    # Which Source 1 records are for learning, early stopping, and measuring.
    # (Based on a fixed hash of the id: the same split on every computer.)
    info = json.loads((split_dir / "features_info.json").read_text())
    sampled = training_records(s1, info["train_fraction"])
    raw_bucket = id_bucket(s1["entity_id"])
    bucket = raw_bucket % 100
    is_holdout = bucket < HOLDOUT_PERCENT
    is_stopping = (bucket >= HOLDOUT_PERCENT) & (bucket < 2 * HOLDOUT_PERCENT)
    # Measure over every sampled record, also the ones without any candidate.
    holdout_rows = np.flatnonzero(sampled & is_holdout)

    pair_group = features["s1_row"].to_numpy()
    holdout = features[is_holdout[pair_group]]
    stopping = features[is_stopping[pair_group]]
    learning = features[~is_holdout[pair_group] & ~is_stopping[pair_group]]
    log(f"learn: {len(learning):,} pairs | early stop: {len(stopping):,} | "
        f"holdout: {len(holdout):,} pairs ({len(holdout_rows):,} Source 1 records)")
    del features, pair_group        # the three parts are copies
    gc.collect()

    stage1_columns = columns
    stage1_models = []
    if stage2:
        # S5: stage 1 on two halves, then the collective features for every part
        stage1_models, probabilities = cross_fit_stage1(learning, stopping, holdout, columns, raw_bucket % 2)
        parts = [learning, stopping, holdout]
        rival_tables = None
        try:
            from rival import load_tables
            rival_tables = load_tables(split_dir)
        except FileNotFoundError as error:
            log(f"rival features skipped (cleaned tables not found: {error})")
        extra = collective_features(
            [(part["s1_row"].to_numpy(), part["s23_row"].to_numpy(), p) for part, p in zip(parts, probabilities)],
            s23["name_core"].to_numpy(), s23["addr_clean"].to_numpy(), rival_tables)
        learning, stopping, holdout = [pd.concat([part.reset_index(drop=True), more], axis=1)
                                       for part, more in zip(parts, extra)]
        del parts, extra
        gc.collect()
        stage1_result = evaluate(None, None, holdout, holdout_rows, truth,
                                 "HOLDOUT, stage 1 alone (mean of both halves)", holdout["p1"].to_numpy())
        columns = columns + STAGE2_COLUMNS
        log("collective features ready")

    # The "LLM layer" (cross_encoder.py): its scores of the unsure pairs, if a GPU notebook made them.
    use_ce = bool(stage1_models) and (split_dir / SCORE_FILE).exists()
    if use_ce:
        scores = pd.read_parquet(split_dir / SCORE_FILE)
        learning, stopping, holdout = [
            pd.concat([part, ce_features(part["s1_row"], part["s23_row"], scores)], axis=1)
            for part in (learning, stopping, holdout)]
        columns = columns + CE_COLUMNS
        log(f"cross-encoder scores of {len(scores):,} pairs added "
            f"({holdout['ce'].notna().mean():.1%} of the holdout pairs have one)")
        del scores

    model, curve = train_model(learning, stopping, columns)
    best = curve.iloc[model.best_iteration - 1]
    log(f"early stopping: stopped after {len(curve)} rounds, best round {model.best_iteration} "
        f"(no better valid log-loss for {EARLY_STOP} rounds)")
    log(f"at the best round: train log-loss {best['train_binary_logloss']:.5f}, "
        f"valid log-loss {best['valid_binary_logloss']:.5f}, train AUC {best['train_auc']:.5f}, "
        f"valid AUC {best['valid_auc']:.5f}")

    result = evaluate(model, columns, holdout, holdout_rows, truth, "HOLDOUT")
    use_stage2 = bool(stage1_models)
    if stage1_models and best_of(stage1_result) > best_of(result):
        log(f"stage 2 did not help ({best_of(result):.4f} vs stage 1 alone {best_of(stage1_result):.4f}): "
            "the final answer uses stage 1 alone")
        use_stage2, result = False, stage1_result
    log("macro F0.5 for each threshold: "
        + "  ".join(f"{t:.2f}:{v:.4f}" for t, v in result["all_thresholds"].items()))
    best_decoder = choose_decoder(result["expected_f05"], result["threshold_f05"])
    best_score = max(result["threshold_f05"], result["expected_f05"])
    log(f"decoder: {best_decoder} (expected-F0.5 must win by more than {DECODER_MARGIN})")

    # S8: where do we lose points?
    chosen = decide(result["scored"], best_decoder, result["threshold"])
    # On Kaggle, files of an earlier run can be read-only links (reused input): replace them.
    out_dir = work_dir / "models"
    for path in [split_dir / "holdout_report.txt", split_dir / "errors.tsv", *out_dir.glob("*")]:
        if path.is_symlink():
            path.unlink()
    report = score_breakdown(s1, holdout_rows, chosen, truth)
    print("\n" + report + "\n", flush=True)
    (split_dir / "holdout_report.txt").write_text(report)
    errors = error_examples(result["scored"], chosen, truth, s1, s23)
    errors.to_csv(split_dir / "errors.tsv", sep="\t", index=False)
    log(f"worst mistakes saved to {split_dir / 'errors.tsv'}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_curve(curve, model.best_iteration, out_dir)
    model.save_model(str(out_dir / "lgbm.txt"), num_iteration=model.best_iteration)
    for k, stage1 in enumerate(stage1_models):
        stage1.save_model(str(out_dir / f"lgbm_stage1_{k}.txt"), num_iteration=stage1.best_iteration)
    (out_dir / "meta.json").write_text(json.dumps({
        "features": columns, "stage1_features": stage1_columns, "use_stage2": use_stage2, "cross_encoder": use_ce and use_stage2,
        "stage1_models": [f"lgbm_stage1_{k}.txt" for k in range(len(stage1_models))], "threshold": result["threshold"], "best_decoder": best_decoder,
        "holdout_macro_f05": best_score, "best_iteration": model.best_iteration}, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps({
        "pairs": {"learn": len(learning), "early_stop": len(stopping), "holdout": len(holdout)},
        "holdout_source1_records": len(holdout_rows),
        "rounds_run": len(curve), "best_round": model.best_iteration, "early_stop_patience": EARLY_STOP,
        "at_best_round": {k: float(v) for k, v in best.items() if k != "round"},
        "holdout": {"auc": result["auc"], "threshold": result["threshold"],
                    "macro_f05_threshold": result["threshold_f05"],
                    "macro_f05_threshold_no_exclusivity": result["no_exclusivity_f05"],
                    "macro_f05_expected": result["expected_f05"], "decoder": best_decoder,
                    "pair_precision": result["precision"], "pair_recall": result["recall"],
                    "macro_f05_by_threshold": result["all_thresholds"]},
    }, indent=2))
    importance = pd.Series(model.feature_importance("gain"), index=columns).sort_values(ascending=False)
    importance.round(0).to_csv(out_dir / "feature_importance.csv", header=["gain"])
    log(f"top features: {', '.join(importance.index[:8])}")
    log(f"saved model to {out_dir}")

    transfer_scores = {}
    if transfer and not stage2:
        transfer_scores = transfer_check(pd.read_parquet(split_dir / "features.parquet"), s1, np.flatnonzero(sampled), truth,
                                         columns, model.best_iteration)
    append_results(work_dir / "results.md", note, best_score, transfer_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--transfer-check", action="store_true")
    parser.add_argument("--note", default="", help="what changed in this run (for results.md)")
    parser.add_argument("--no-stage2", action="store_true", help="one model only (no collective features)")
    args = parser.parse_args()
    run_training(args.work_dir, args.data_dir, args.transfer_check, args.note, stage2=not args.no_stage2)


if __name__ == "__main__":
    main()

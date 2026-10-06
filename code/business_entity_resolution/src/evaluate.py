"""
S8 - Scoring, exactly like the official rules, plus reports that show WHERE we lose.

Macro F0.5: compute F0.5 for every Source 1 record, then take the average.
  - no true matches and we predict nothing  -> 1.0
  - no true matches but we predict anything -> 0.0
  - otherwise F0.5 = 1.25 * P * R / (0.25 * P + R)

All pair tables here use row numbers of the cleaned tables: columns s1_row, s23_row.
"""
import numpy as np
import pandas as pd

PAIR = ["s1_row", "s23_row"]


def f05(n_predicted, n_true, n_correct):
    """F0.5 of one Source 1 record, from three counts (works on arrays too)."""
    n_predicted = np.asarray(n_predicted, dtype=float)
    n_true = np.asarray(n_true, dtype=float)
    n_correct = np.asarray(n_correct, dtype=float)

    precision = np.divide(n_correct, n_predicted, out=np.zeros_like(n_correct), where=n_predicted > 0)
    recall = np.divide(n_correct, n_true, out=np.zeros_like(n_correct), where=n_true > 0)
    bottom = 0.25 * precision + recall
    score = np.divide(1.25 * precision * recall, bottom, out=np.zeros_like(bottom), where=bottom > 0)

    both_empty = (n_true == 0) & (n_predicted == 0)
    return np.where(both_empty, 1.0, score)


def count_per_s1(s1_rows, pairs):
    """How many pairs each Source 1 row has (0 when it has none)."""
    counts = pairs.groupby("s1_row").size()
    return pd.Series(np.asarray(s1_rows)).map(counts).fillna(0).to_numpy()


def per_record_counts(s1_rows, predicted, truth):
    """Three counts per Source 1 row: predicted, true, correct."""
    predicted = predicted[PAIR].drop_duplicates()
    truth = truth[truth["s1_row"].isin(s1_rows)]
    correct = predicted.merge(truth[PAIR], on=PAIR)
    return (count_per_s1(s1_rows, predicted),
            count_per_s1(s1_rows, truth),
            count_per_s1(s1_rows, correct))


def macro_f05(s1_rows, predicted, truth):
    """Official score over the given Source 1 rows (singletons included)."""
    return float(f05(*per_record_counts(s1_rows, predicted, truth)).mean())


# ---------------------------------------------------------------------------
# S2 report: how good is the candidate list?
# ---------------------------------------------------------------------------
def blocking_report(s1, candidates, truth, query_rows, s23=None):
    """Readable text report of candidate quality.

    s1         : Source 1 table (to know the country of each row)
    candidates : output of blocking.build_candidates
    truth      : true pairs as rows (io_utils.truth_as_rows)
    query_rows : the Source 1 rows we produced candidates for
    s23        : optional Source 2/3 table, for the reduction ratio
    """
    truth = truth[truth["s1_row"].isin(query_rows)]
    found = truth.merge(candidates, on=PAIR, how="inner")
    n_found = count_per_s1(query_rows, found)
    n_true = count_per_s1(query_rows, truth)

    lines = []
    add = lines.append
    add(f"Source 1 records checked     : {len(query_rows):,}")
    add(f"candidates per record (avg)  : {len(candidates) / max(len(query_rows), 1):.1f}")
    if s23 is not None:
        # all pairs the search could have compared (same country only)
        queries_per_country = s1["country"].iloc[query_rows].value_counts()
        docs_per_country = s23["country"].value_counts()
        all_pairs = (queries_per_country * docs_per_country.reindex(queries_per_country.index).fillna(0)).sum()
        add(f"reduction ratio              : {1 - len(candidates) / max(all_pairs, 1):.6f}   "
            f"({len(candidates):,} of {all_pairs:,.0f} same-country pairs)")
    add(f"true pairs                   : {len(truth):,}")
    add(f"PAIR RECALL                  : {len(found) / max(len(truth), 1):.4f}   (target >= 0.97)")
    with_matches = n_true > 0
    add(f"records with ALL matches found: {(n_found[with_matches] == n_true[with_matches]).mean():.4f}")
    ceiling = f05(n_found, n_true, n_found).mean()
    add(f"best possible macro F0.5     : {ceiling:.4f}   (if the model were perfect on these candidates)")

    add("")
    add("recall if we kept only the top K candidates (the rescued pairs, rank -1, not counted):")
    for k in (1, 3, 5, 10, 15, 20, 30, 50):
        add(f"   K={k:<3} {((found['candidate_rank'] >= 0) & (found['candidate_rank'] < k)).sum() / max(len(truth), 1):.4f}")
    rescued = found[found["candidate_rank"] < 0]
    if len(rescued):
        add(f"   +rescued pairs (candidate_rank -1, n={len(rescued):,}): "
            f"pair recall {len(found) / max(len(truth), 1):.4f}")

    add("")
    add("true pairs found by each method (a pair can be found by several):")
    for column in sorted(c for c in candidates.columns if c.startswith("score_")):
        only_this = found[column].notna() & (found["n_methods"] == 1)
        add(f"   {column[6:]:<16} {found[column].notna().mean():.4f}   found ONLY by this: {only_this.sum():,}")

    add("")
    add("pair recall by country:")
    country = s1["country"].to_numpy()
    truth_country = pd.Series(country[truth["s1_row"].to_numpy()])
    found_country = pd.Series(country[found["s1_row"].to_numpy()])
    for name in sorted(truth_country.unique()):
        total = (truth_country == name).sum()
        add(f"   {name:<10} {(found_country == name).sum() / total:.4f}   ({total:,} true pairs)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# S8 report: where does the final answer lose points?
# ---------------------------------------------------------------------------
def score_breakdown(s1, s1_rows, predicted, truth):
    """Macro F0.5, precision and recall, split by country, number of true
    matches, and whether the Source 1 address is missing."""
    n_pred, n_true, n_correct = per_record_counts(s1_rows, predicted, truth)
    rows = np.asarray(s1_rows)
    table = pd.DataFrame({
        "f05": f05(n_pred, n_true, n_correct),
        "country": s1["country"].to_numpy()[rows],
        "true_matches": pd.cut(n_true, [-1, 0, 1, 3, np.inf], labels=["0", "1", "2-3", "4+"]),
        "address_missing": np.where(s1["addr_clean"].to_numpy()[rows] == "", "yes", "no"),
    })

    lines = [f"macro F0.5 : {table['f05'].mean():.4f}   over {len(table):,} Source 1 records",
             f"precision  : {n_correct.sum() / max(n_pred.sum(), 1):.4f}   (all pairs together)",
             f"recall     : {n_correct.sum() / max(n_true.sum(), 1):.4f}"]
    for column in ("country", "true_matches", "address_missing"):
        lines.append(f"\nby {column}:")
        groups = table.groupby(column, observed=True)["f05"].agg(["mean", "size"])
        for name, row in groups.iterrows():
            lines.append(f"   {str(name):<8} {row['mean']:.4f}   ({int(row['size']):,} records)")
    return "\n".join(lines)


def error_examples(scored, predicted, truth, s1, s23, n=50):
    """The worst mistakes, with the texts, to read and learn from.

    scored    : all candidate pairs of the checked records with a 'probability' column
    predicted : the pairs we answered with
    Returns a table: kind (false_positive / false_negative / not_in_candidates), probability, texts.
    """
    s1_rows = scored["s1_row"].unique()
    truth = truth[truth["s1_row"].isin(s1_rows)].assign(is_true=True)
    answered = predicted[PAIR].assign(answered=True)
    pairs = (scored.merge(truth, on=PAIR, how="outer")
                   .merge(answered, on=PAIR, how="left"))
    pairs[["is_true", "answered"]] = pairs[["is_true", "answered"]].astype("boolean").fillna(False)

    false_pos = pairs[pairs["answered"] & ~pairs["is_true"]].nlargest(n, "probability")
    false_neg = pairs[~pairs["answered"] & pairs["is_true"] & pairs["probability"].notna()]
    false_neg = false_neg.nsmallest(n, "probability")
    missed = pairs[pairs["is_true"] & pairs["probability"].isna()].head(n)

    errors = pd.concat([false_pos.assign(kind="false_positive"),
                        false_neg.assign(kind="false_negative"),
                        missed.assign(kind="not_in_candidates")], ignore_index=True)
    a, b = errors["s1_row"].to_numpy(), errors["s23_row"].to_numpy()
    return pd.DataFrame({
        "kind": errors["kind"],
        "probability": errors["probability"].round(3),
        "s1_name": s1["business_name"].to_numpy()[a],
        "s23_name": s23["business_name"].to_numpy()[b],
        "s1_address": s1["business_address"].to_numpy()[a],
        "s23_address": s23["business_address"].to_numpy()[b],
        "country": s1["country"].to_numpy()[a],
        "s1_id": s1["entity_id"].to_numpy()[a],
        "s23_id": s23["entity_id"].to_numpy()[b],
    })

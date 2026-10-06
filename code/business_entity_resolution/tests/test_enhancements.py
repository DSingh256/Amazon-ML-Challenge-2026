"""
Tests of the model enhancements (rival features, rescue candidates, gap features).

Run:  python tests/test_enhancements.py      (or: pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from blocking import combine_and_trim, combine_and_trim_in_pieces  # noqa: E402
from rival import RIVAL_COLUMNS, digit_key, name_key, plain, rival_features, tables_from  # noqa: E402
from stage2 import STAGE2_COLUMNS, group_numbers  # noqa: E402


def hits_table():
    """Two Source 1 records, a handful of hits each (ranks 0..2 per record and direction).
    A fresh DataFrame every call: combine_and_trim_in_pieces may change dtypes in place."""
    rows = []
    for i, (a, b, rank, score) in enumerate([
            (0, 10, 0, 0.9), (0, 11, 1, 0.8), (0, 12, 2, 0.7), (0, 13, 3, 0.6), (0, 14, 4, 0.55),
            (1, 20, 0, 0.85), (1, 21, 1, 0.75), (1, 22, 2, 0.65)]):
        rows.append({"s1_row": a, "s23_row": b, "method": "name_addr_fwd", "score": score, "rank": rank})
    return pd.DataFrame(rows)


def test_no_rescue_same_columns():
    candidates, rescued = combine_and_trim(hits_table(), 2, rescue=0)
    assert rescued is None
    assert set(candidates["s1_row"]) == {0, 1}
    assert (candidates["candidate_rank"] == [0, 1, 0, 1]).all()


def test_rescue_rows_directly_below_the_last_kept():
    candidates, rescued = combine_and_trim(hits_table(), 2, rescue=2)
    assert rescued is not None
    # record 0: hits (12, 0.7) and (13, 0.6) are ranks 2 and 3 = the first two below the last
    # kept candidate (11); hit (14, 0.55) is the third -> not rescued with rescue=2
    assert set(zip(rescued["s1_row"], rescued["s23_row"])) == {(0, 12), (0, 13), (1, 22)}
    assert (rescued["candidate_rank"] == -1).all()
    # rescue rows keep the same columns as candidates (same schema)
    assert list(rescued.columns) == list(candidates.columns)
    # search_margin is computed later, in combine_and_trim_in_pieces (per piece)


def test_rescue_also_with_max_candidates_one():
    """Rescue works for any max_candidates: the pairs below the last kept one, however few."""
    candidates, rescued = combine_and_trim(hits_table(), 1, rescue=2)
    assert rescued is not None
    assert set(zip(rescued["s1_row"], rescued["s23_row"])) == {(0, 11), (0, 12), (1, 21), (1, 22)}


def test_search_margin_of_the_piece_step():
    """combine_and_trim_in_pieces adds search_margin: NaN for kept candidates, the gap down
    from the record's best kept candidate for rescue rows."""
    table = combine_and_trim_in_pieces(hits_table(), 2, rescue=2)
    assert "search_margin" in table.columns
    by_pair = {(int(a), int(b)): m for a, b, m in
               zip(table["s1_row"], table["s23_row"], table["search_margin"])}
    # record 0: best kept score 0.9 -> rescue row (0, 12) sits 0.2 below it, (0, 13) 0.3
    assert np.isnan(by_pair[(0, 10)]) and np.isnan(by_pair[(0, 11)])
    assert np.isclose(by_pair[(0, 12)], 0.2) and np.isclose(by_pair[(0, 13)], 0.3)
    assert np.isclose(by_pair[(1, 22)], 0.2)     # record 1 best kept 0.85, copy 0.65
    # 4 kept + 3 rescued = 7 rows (record 0 has 5 hits, record 1 has 3)
    assert len(table) == 7


def hits_with_reverse():
    """A record whose hits come from two retrievers (like fwd + rev)."""
    fwd = hits_table()
    rev = pd.DataFrame([
        {"s1_row": 0, "s23_row": 11, "method": "name_addr_rev", "score": 0.95, "rank": 0},
        {"s1_row": 0, "s23_row": 12, "method": "name_addr_rev", "score": 0.5, "rank": 1}])
    return pd.concat([fwd, rev], ignore_index=True)


def test_rescue_never_duplicates_a_candidate():
    candidates, rescued = combine_and_trim(hits_with_reverse(), 2, rescue=2)
    kept_pairs = set(zip(candidates["s1_row"], candidates["s23_row"]))
    if rescued is not None:
        assert kept_pairs.isdisjoint(set(zip(rescued["s1_row"], rescued["s23_row"])))


def rival_tables():
    s1 = pd.DataFrame({
        "business_name": ["Alpha Trading Ltd", "Alpha Trading Inc", "Beta Stores"],
        "business_address": ["12 High Street", "9 Market Road", "5 South Avenue"]})
    s23 = pd.DataFrame({
        "business_name": ["alpha trading private limited", "beta stores", "gamma tech llc"],
        "business_address": ["", "5 south avenue", ""]})
    return tables_from(s1, s23)


def test_rival_covers_all_columns_inside_the_window_only():
    tables = rival_tables()
    s1_row = np.array([0, 1, 2], dtype=np.int64)     # the three S2/3 copies in order
    s23_row = np.array([0, 1, 2], dtype=np.int64)
    p = np.array([0.5, 0.999, 0.0001], dtype=np.float64)   # pair 0 unsure, 1 above HIGH, 2 below LOW
    out = rival_features(s1_row, s23_row, p, tables)
    assert sorted(out) == sorted(RIVAL_COLUMNS)
    for column in RIVAL_COLUMNS:
        values = out[column]
        assert len(values) == 3
        assert np.isnan(values[2]), f"{column} must be NaN below the unsure window"
        assert np.isnan(values[1]), f"{column} must be NaN above the unsure window"
    assert not np.isnan(out["rv_sim"][0])
    assert out["rv_empty"][0] == 1.0                  # the copy has an empty address
    # 'alpha trading' has two Source 1 rivals (Ltd and Inc)
    assert out["rv_n2"][0] == 2.0
    # the exact legal form (ltd vs inc) is the tie-breaker the features exist for
    assert out["rv_n_better"][0] >= 0


def test_name_key_and_digit_key():
    pl = plain(["Alpha Trading Pvt. Ltd. (id:123456)", "www.beta.com"])
    # legal words, ids and long numbers go; a name that is ONLY a website becomes ""
    assert name_key(pl).tolist() == ["alpha trading", ""]
    assert digit_key(plain(["12 High St, 12", "no numbers here"])).tolist() == ["12", ""]


def test_group_numbers_basics():
    s1_row = np.array([0, 0, 1], dtype=np.int64)
    s23_row = np.array([10, 11, 20], dtype=np.int64)
    p = np.array([0.9, 0.3, 0.5], dtype=np.float64)
    order, group_id, q, numbers = group_numbers(s1_row, s23_row, p)
    for column in ("p1", "p1_rank", "p1_others_max", "p1_others_sum", "n_sure_others"):
        assert column in numbers and column in STAGE2_COLUMNS
    # rank 1 for the best candidate of each record
    assert sorted(numbers["p1_rank"].tolist()) == [1, 1, 2]
    # the second candidate of record 0 sees the other candidate's probability
    assert np.isclose(numbers["p1_others_max"][1], 0.9)


def test_threshold_grid():
    import train
    assert len(train.THRESHOLDS) == 13 and 0.30 in np.asarray(train.THRESHOLDS) and 0.90 in np.asarray(train.THRESHOLDS)


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_") and callable(func):
            func()
            print(f"ok  {name}")
    print("all enhancement tests passed")

"""
S5 - Second stage: "collective" features (look at the other candidates of the same record).

The first model (stage 1) judges every pair on its own. But the candidates of one
Source 1 record are not independent: a business usually has 3-4 records in Source 2/3,
and they look like EACH OTHER (same address written the same way, same trade name).
Errors we saw on the holdout:
  - true match with a different (trade / garbled) name, but the same address as the
    other, sure matches of the record -> stage 1 says 0.00
  - look-alike record that is not like the sure matches of the record
Stage 2 therefore adds, for every pair (a, b), numbers computed from the stage-1
probabilities p of ALL candidates c of the same Source 1 record a:
  p1                 : stage-1 probability of this pair
  p1_rank            : rank of p1 inside the record (1 = best)
  p1_others_max/sum  : best / total probability of the OTHER candidates
  n_sure_others      : how many other candidates have p > 0.5
  sib_addr_support   : sum_c p_c * cos(address b, address c) / sum_c p_c   (c != b)
  sib_name_support   : the same with the names
("collective entity resolution": Bhattacharya & Getoor, ACM TKDD 2007.)

To learn stage 2 without cheating, stage 1 is trained twice on two halves of the
training records ("cross-fitting"); each half gets probabilities from the model that
never saw it. The holdout and the test pairs get the mean of both models.

The cosine sums need no pair-by-pair loop: per record we add up the p-weighted
3-letter-piece vectors of all its candidates (one sparse matrix product), then read
the entries of that sum at the pieces of b.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from features import N_HASHED_PIECES, char_tfidf, rank_in_group
from rival import RIVAL_COLUMNS, rival_features

STAGE2_COLUMNS = ["p1", "p1_rank", "p1_others_max", "p1_others_sum", "n_sure_others",
                  "sib_addr_support", "sib_name_support"] + RIVAL_COLUMNS
SURE = 0.5
PAIRS_PER_STEP = 300_000


def text_vectors(texts):
    """Rows scaled to length 1 of 3-letter-piece TF-IDF (hashed; see features.char_tfidf)."""
    return char_tfidf(pd.Series([], dtype=object), texts.reset_index(drop=True))[1]


def weighted_cosine_sums(group, vector_rows, p, vectors):
    """For every pair i: sum over pairs j of the same group of p[j] * cos(i, j), without j = i.
    group must be sorted (pairs of one Source 1 record next to each other)."""
    out = np.zeros(len(group), dtype=np.float32)
    starts = np.flatnonzero(np.r_[True, group[1:] != group[:-1]])
    bounds = np.r_[starts[::max(1, len(starts) * PAIRS_PER_STEP // max(len(group), 1))], len(group)]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        v = vectors[vector_rows[lo:hi]]
        v.sort_indices()
        local = (group[lo:hi] - group[lo]).astype(np.int64)
        n_groups = int(local[-1]) + 1
        weights = sp.csr_matrix((p[lo:hi].astype(np.float32), (local, np.arange(hi - lo))),
                                shape=(n_groups, hi - lo))
        total = (weights @ v).tocsr()
        total.sort_indices()
        total_keys = (np.repeat(np.arange(n_groups, dtype=np.int64), np.diff(total.indptr)) * N_HASHED_PIECES
                      + total.indices)
        pair_of_value = np.repeat(np.arange(hi - lo), np.diff(v.indptr))
        wanted = local[pair_of_value] * N_HASHED_PIECES + v.indices
        where = np.minimum(np.searchsorted(total_keys, wanted), max(len(total_keys) - 1, 0))
        found = np.where(total_keys[where] == wanted, total.data[where], 0) if len(total_keys) else 0
        dot = np.bincount(pair_of_value, weights=v.data * found, minlength=hi - lo)
        self_part = p[lo:hi] * np.bincount(pair_of_value, weights=v.data ** 2, minlength=hi - lo)
        out[lo:hi] = dot - self_part
    return out


def group_numbers(s1_row, s23_row, p):
    """Sort one part's pairs by record; return (order, group id, sorted p, dict of the
    probability features in sorted order)."""
    order = np.lexsort((s23_row, s1_row))
    group = s1_row[order]
    q = p[order].astype(np.float64)
    group_id = np.cumsum(np.r_[True, group[1:] != group[:-1]]) - 1
    group_sum = np.bincount(group_id, weights=q)[group_id]
    sure = (q > SURE).astype(np.float64)
    rank = rank_in_group([group_id], q)
    top1 = pd.Series(q).groupby(group_id).transform("max").to_numpy()
    second = pd.Series(np.where(rank == 1, -1.0, q)).groupby(group_id).transform("max").to_numpy()
    others_max = np.where(rank == 1, second, top1)
    others_max[others_max < 0] = np.nan                  # the record has only this candidate
    numbers = {"p1": q, "p1_rank": rank, "p1_others_max": others_max,
               "p1_others_sum": group_sum - q,
               "n_sure_others": np.bincount(group_id, weights=sure)[group_id] - sure}
    return order, group_id, q, numbers


def collective_features(parts, s23_names, s23_addresses, rival_tables=None):
    """Stage-2 features for several independent parts (e.g. learn / early-stop / holdout).
    parts : list of (s1_row, s23_row, p) arrays, one entry per candidate pair; a part must
            hold ALL candidates of each of its Source 1 records
    s23_names / s23_addresses : name_core / addr_clean of the Source 2/3 table (by row number)
    rival_tables : optional tables of the split (rival.tables_from / load_tables); when given,
            the same-name / same-address rival features (rival.py) are added to every part.
    Returns one DataFrame (columns STAGE2_COLUMNS) per part, in the input row order.
    The text vectors are made once for all parts, one field at a time (memory)."""
    prepared = [group_numbers(*part) for part in parts]
    used = np.unique(np.concatenate([part[1] for part in parts]))
    for name, texts in (("addr", s23_addresses), ("name", s23_names)):
        texts = pd.Series(np.asarray(texts, dtype=object)[used]).fillna("")
        empty = texts.to_numpy() == ""
        vectors = text_vectors(texts)
        for (s1_row, s23_row, _), (order, group_id, q, numbers) in zip(parts, prepared):
            rows = np.searchsorted(used, s23_row[order])
            sums = weighted_cosine_sums(group_id, rows, q, vectors)
            others = numbers["p1_others_sum"]
            support = np.divide(sums, others, out=np.full(len(q), np.nan), where=others > 1e-3)
            support[empty[rows]] = np.nan
            numbers[f"sib_{name}_support"] = support
        del vectors
    if rival_tables is not None:
        for (s1_row, s23_row, p), (order, _, _, numbers) in zip(parts, prepared):
            numbers.update(rival_features(s1_row[order], s23_row[order], p[order], rival_tables))
    frames = []
    for order, _, _, numbers in prepared:
        # reindex: rival columns are missing (NaN) when rival_tables is None
        frame = pd.DataFrame({k: np.asarray(numbers[k], dtype=np.float32)
                              for k in STAGE2_COLUMNS}).reindex(columns=STAGE2_COLUMNS)
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        frames.append(frame.iloc[back].reset_index(drop=True))
    return frames

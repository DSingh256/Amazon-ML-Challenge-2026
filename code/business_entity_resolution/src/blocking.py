"""
S2 - Blocking: for every Source 1 record, find a SHORT list of Source 2/3
records that could be the same business ("candidates").

Comparing every Source 1 record with every Source 2/3 record would take
1.7 million x 10 million comparisons - impossible. Instead:

  1. Turn each text into a vector of 3-letter pieces (TF-IDF).
     "kenwood" -> "ken", "enw", "nwo", "woo", "ood". Rare pieces get a high
     weight, common pieces a low weight.
  2. Two texts are similar when they share many high-weight pieces.
     A sparse matrix multiplication gives the similarity of one query with
     ALL records at once; we keep only the top k.
  3. Speed: a query only uses its rarest pieces, and pieces that appear in a
     big share of all documents are skipped (see config.py). The work of a
     query is the total length of the document lists of its pieces, so
     skipping common pieces removes most of the work.
  4. We search in both directions:
        forward : Source 1 record  -> its best Source 2 and Source 3 records
        reverse : Source 2/3 record -> its best Source 1 records
     Reverse search helps when a Source 2/3 record is short
     (e.g. only "WB, Howrah, Astha Apartment").
  5. We only compare records with the same `country` value. This is a
     generic group-by: any new country (France) is handled the same way.
  6. All hits are merged and each Source 1 keeps its best `max_candidates`.

Main function: build_candidates(s1, s23, config)
Output uses row numbers of the cleaned tables (s1_row, s23_row), not ids:
ids are long strings and would need many GB at full size.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from io_utils import log

try:                                    # fast, multi-threaded top-k (Apache-2.0)
    from sparse_dot_topn import sp_matmul_topn
except ImportError:                     # plain scipy fallback (slower)
    sp_matmul_topn = None


# ---------------------------------------------------------------------------
# Sparse-matrix helpers
# ---------------------------------------------------------------------------
def top_k_per_row(matrix, k):
    """For each row of a sparse matrix, find the k largest values.

    Returns four arrays of equal length: row, column, value, rank
    (rank 0 = largest value of that row).
    """
    matrix = sp.csr_matrix(matrix)
    rows = np.repeat(np.arange(matrix.shape[0]), np.diff(matrix.indptr))

    # Sort all values: first by row, inside a row from high to low.
    # One sort key is much faster than sorting by two keys: the whole part is
    # the row number, the fraction is (1 - value) so bigger values sort first.
    # (Values are between 0 and 1.)
    sort_key = rows + (1.0 - matrix.data.astype(np.float64))
    order = np.argsort(sort_key, kind="stable")
    sorted_rows = rows[order]

    # Position inside its own row = position in the sorted list - start of the row.
    rank = np.arange(len(order)) - matrix.indptr[sorted_rows]
    keep = rank < k
    return (sorted_rows[keep],
            matrix.indices[order][keep],
            matrix.data[order][keep],
            rank[keep])


def pick_query_ngrams(queries, doc_freq, n_docs, config, block_rows=200_000):
    """Keep only the useful pieces of every query:
        - at most `max_query_ngrams` of its highest-weighted (= rarest) pieces;
        - a piece found in more than `max_ngram_share` of the documents is dropped,
          except that the `min_query_ngrams` rarest pieces are always kept.
    Done in blocks of rows: sorting all pieces of millions of queries at once needs
    far more memory than the machine has (this killed the first test-prep run).
    """
    max_df = max(50, config.max_ngram_share * n_docs)
    blocks = []
    for start in range(0, queries.shape[0], block_rows):
        part = queries[start:start + block_rows]
        rows, cols, values, rank = top_k_per_row(part, config.max_query_ngrams)
        keep = (rank < config.min_query_ngrams) | (doc_freq[cols] <= max_df)
        blocks.append(sp.csr_matrix((values[keep], (rows[keep], cols[keep])),
                                    shape=part.shape, dtype=np.float32))
    return sp.vstack(blocks, format="csr", dtype=np.float32)


def field_patterns(vectorizer, table, columns):
    """One 0/1 matrix per text field: which pieces come from that field of every record."""
    patterns = []
    for column in columns:
        matrix = vectorizer.transform(table[column])
        matrix.data[:] = 1
        patterns.append(matrix)
    return patterns


def pick_query_ngrams_by_field(queries, patterns, doc_freq, n_docs, config, block_rows=200_000):
    """Like pick_query_ngrams, but the pieces are chosen separately inside every field
    (name / address). Choosing the rarest pieces of the whole text let the rare address
    words crowd out the name, so a record whose other copy has no address (or a different
    one) was never found. Every field gets its own share (config.field_quota) of the
    max_query_ngrams; a field that is short or missing leaves its room to the other one."""
    max_df = max(50, config.max_ngram_share * n_docs)
    total, floor = config.max_query_ngrams, config.min_query_ngrams
    blocks = []
    for start in range(0, queries.shape[0], block_rows):
        part = queries[start:start + block_rows]
        fields = [part.multiply(pattern[start:start + block_rows]).tocsr() for pattern in patterns]
        counts = [np.diff(field.indptr) for field in fields]
        selected = None
        for i, field in enumerate(fields):
            other = sum(counts) - counts[i]
            quota = np.maximum(config.field_quota[i], total - other)          # own share, or all the free room
            own_floor = np.maximum(floor // len(fields), floor - other)       # rarest pieces that are always kept
            rows, cols, values, rank = top_k_per_row(field, total)
            keep = (rank < quota[rows]) & ((rank < own_floor[rows]) | (doc_freq[cols] <= max_df))
            picked = sp.csr_matrix((values[keep], (rows[keep], cols[keep])), shape=part.shape, dtype=np.float32)
            selected = picked if selected is None else selected.maximum(picked)
        blocks.append(selected)
    return sp.vstack(blocks, format="csr", dtype=np.float32)


def split_into_chunks(work_per_row, budget):
    """Cut rows into consecutive chunks whose total work stays under budget.
    A single very expensive row still gets its own chunk."""
    total = np.cumsum(work_per_row)
    chunks, start = [], 0
    while start < len(work_per_row):
        already_done = total[start - 1] if start > 0 else 0
        stop = int(np.searchsorted(total, already_done + budget, side="right"))
        stop = max(stop, start + 1)
        chunks.append((start, stop))
        start = stop
    return chunks


def top_k_of_product(queries, doc_blocks, k, n_threads):
    """Top k documents of every query row of (queries @ documents).
    doc_blocks: [(first document number, piece-by-document matrix of a block of documents)].
    Each block is searched on its own and the per-block top k lists are merged; this gives
    exactly the same result and is faster for millions of documents (the work area of
    one block fits in the CPU cache)."""
    rows, cols, values = [], [], []
    for offset, block in doc_blocks:
        if sp_matmul_topn is not None:
            result = sp_matmul_topn(queries, block, top_n=k, sort=True, n_threads=n_threads)
            rows.append(np.repeat(np.arange(result.shape[0]), np.diff(result.indptr)))
            cols.append(result.indices + offset)
            values.append(result.data)
        else:
            r, c, v, _ = top_k_per_row(queries @ block, k)
            rows.append(r), cols.append(c + offset), values.append(v)
    n_docs = sum(block.shape[1] for _, block in doc_blocks)
    merged = sp.csr_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(queries.shape[0], n_docs))
    return top_k_per_row(merged, k)


def cached_search(cache_file, queries, documents, k, config, dry_run=False, patterns=None, coverage=0.0):
    """search(), but the result is saved in cache_file and reused by a later run
    (a search of a big country takes 1-3 hours)."""
    if cache_file is not None and not dry_run and cache_file.exists():
        log(f"      reusing the saved search {cache_file.name}")
        return pd.read_parquet(cache_file)
    found = search(queries, documents, k, config, dry_run, patterns, coverage)
    if cache_file is not None and not dry_run:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        found.to_parquet(cache_file)
        log(f"      saved {cache_file.name}")
    return found


def search(queries, documents, k, config, dry_run=False, patterns=None, coverage=0.0):
    """For every query row, find the k documents with the highest similarity.

    queries, documents : sparse TF-IDF matrices (rows = records)
    dry_run            : only print how much work the search would be
    patterns           : per text field, which pieces of every query come from it (see
                         pick_query_ngrams_by_field); None = rarest pieces of the whole text
    coverage           : 0 = cosine similarity (a long document is penalised). Above 0 the score
                         is `coverage` x (share of the query pieces that the document contains)
                         + (1 - coverage) x cosine, so a short query still finds the long record
                         it was cut from (a Source 2/3 text without the address of Source 1).
    Returns a DataFrame with columns: query, doc, score, rank
    """
    empty = pd.DataFrame({"query": [], "doc": [], "score": [], "rank": []})
    if queries.shape[0] == 0 or documents.shape[0] == 0:
        return empty

    documents = sp.csr_matrix(documents, dtype=np.float32)
    n_docs = documents.shape[0]
    doc_freq = np.bincount(documents.indices, minlength=documents.shape[1])
    if patterns:
        queries = pick_query_ngrams_by_field(queries, patterns, doc_freq, n_docs, config)
    else:
        queries = pick_query_ngrams(queries, doc_freq, n_docs, config)

    if coverage > 0:
        weight_sum = np.asarray(queries.sum(axis=1)).ravel()
        queries = sp.diags(1.0 / np.maximum(weight_sum, 1e-6)).dot(queries).tocsr().astype(np.float32)
        documents = documents.copy()
        documents.data = (coverage + (1.0 - coverage) * documents.data).astype(np.float32)

    # Work for one query = sum of the list lengths of the pieces it uses.
    has_piece = queries.copy()
    has_piece.data[:] = 1
    work_per_query = has_piece @ doc_freq

    log(f"      search: {queries.shape[0]:,} queries, {documents.shape[0]:,} docs, "
        f"total work {work_per_query.sum() / 1e6:,.0f} million")
    if dry_run:
        return empty

    # For each 3-letter piece: the list of documents that contain it (per block of documents).
    block = config.doc_block_size or n_docs
    doc_blocks = [(first, sp.csr_matrix(documents[first:first + block].T, dtype=np.float32))
                  for first in range(0, n_docs, block)]
    parts = []
    chunks = split_into_chunks(work_per_query, config.work_budget)
    for i, (start, stop) in enumerate(chunks):
        q, d, s, r = top_k_of_product(queries[start:stop], doc_blocks, k, config.n_threads)
        parts.append(pd.DataFrame({"query": q + start, "doc": d, "score": s, "rank": r}))
        if (i + 1) % 20 == 0:
            log(f"      chunk {i + 1}/{len(chunks)}")
    return pd.concat(parts, ignore_index=True) if parts else empty


# ---------------------------------------------------------------------------
# One country
# ---------------------------------------------------------------------------
def run_retriever(s1_group, s23_group, retriever, config, is_query, dry_run=False, cache_prefix=None):
    """Run one retriever (e.g. "name_addr") inside one country.

    s1_group, s23_group : records of this country (index = global row number)
    is_query            : True for Source 1 records we want candidates for
    cache_prefix        : Path start of the files where each search is saved (None = no saving)
    Returns hits: s1_row, s23_row, method, score, rank
    """
    column = retriever.text_column
    log(f"    retriever '{retriever.name}'")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=config.ngram_range,
                                 sublinear_tf=True, dtype=np.float32)
    vectorizer.fit(pd.concat([s1_group[column], s23_group[column]]))
    s1_matrix = vectorizer.transform(s1_group[column])
    s23_matrix = vectorizer.transform(s23_group[column])

    patterns1 = field_patterns(vectorizer, s1_group, retriever.fields)
    patterns23 = field_patterns(vectorizer, s23_group, retriever.fields)

    s1_rows = s1_group.index.to_numpy()
    s23_rows = s23_group.index.to_numpy()
    hits = []

    # Forward: Source 1 -> best records of Source 2, then of Source 3.
    query_positions = np.flatnonzero(is_query)
    for source in (2, 3):
        doc_positions = np.flatnonzero(s23_group["source"].to_numpy() == source)
        cache = None if cache_prefix is None else Path(f"{cache_prefix}_fwd{source}.parquet")
        found = cached_search(cache, s1_matrix[query_positions], s23_matrix[doc_positions],
                              retriever.k_forward, config, dry_run,
                              [p[query_positions] for p in patterns1])
        hits.append(pd.DataFrame({
            "s1_row": s1_rows[query_positions[found["query"].astype(int)]],
            "s23_row": s23_rows[doc_positions[found["doc"].astype(int)]],
            "method": f"{retriever.name}_fwd",
            "score": found["score"],
            "rank": found["rank"],
        }))

    # Reverse: every Source 2/3 record -> its best Source 1 records.
    if retriever.k_reverse > 0:
        cache = None if cache_prefix is None else Path(f"{cache_prefix}_rev.parquet")
        found = cached_search(cache, s23_matrix, s1_matrix, retriever.k_reverse, config, dry_run, patterns23,
                              config.reverse_coverage)
        found = found[is_query[found["doc"].astype(int)]]     # only Source 1 we answer for
        hits.append(pd.DataFrame({
            "s1_row": s1_rows[found["doc"].astype(int)],
            "s23_row": s23_rows[found["query"].astype(int)],
            "method": f"{retriever.name}_rev",
            "score": found["score"],
            "rank": found["rank"],
        }))

    return pd.concat(hits, ignore_index=True)


def combine_and_trim(hits, max_candidates, rescue=0):
    """Merge hits of all retrievers into one row per (s1_row, s23_row) pair
    and keep the best `max_candidates` pairs of each Source 1 record.

    "Best" = found at the best rank by any retriever; ties broken by score.
    Every retriever's score is kept as its own column (used later as features).
    Written with plain arrays: tens of millions of hits in a big country would need
    many GB with a pandas pivot table.

    Unsearched-best rescue (`rescue` > 0): a true pair whose text is very different
    (native-script name, address reduced to a city) often scores just below the last
    candidate the top-k search kept. Its pair IS in the raw search result (only the kept
    top-k lists are not), so the best-scoring non-candidate pairs directly below each
    record's last kept candidate are added too — no new search. The model decides, guided
    by the `search_margin` feature (features.py): a copy is never the worst of its record.

    Returns two DataFrames with the same columns: the candidates (candidate_rank 0..max-1)
    and the rescue rows (candidate_rank = -1).
    """
    rescue_n = max(0, int(rescue))
    s1_row = hits["s1_row"].to_numpy(np.int64)
    s23_row = hits["s23_row"].to_numpy(np.int64)
    pair, first_rows = pd.factorize(s1_row * (1 << 32) + s23_row)     # pair number of every hit
    n_pairs = len(first_rows)
    method, method_names = pd.factorize(hits["method"], sort=True)
    best_rank = pd.Series(hits["rank"].to_numpy()).groupby(pair).min().to_numpy()
    best_score = pd.Series(hits["score"].to_numpy()).groupby(pair).max().to_numpy()
    pair_method = np.unique(pair.astype(np.int64) * len(method_names) + method)
    n_methods = np.bincount(pair_method // len(method_names), minlength=n_pairs)
    method_scores = {}
    for j, name in enumerate(method_names):
        found = method == j
        score = np.full(n_pairs, np.nan, dtype=np.float32)
        best = pd.Series(hits["score"].to_numpy()[found]).groupby(pair[found]).max()
        score[best.index.to_numpy()] = best.to_numpy()
        method_scores[f"score_{name}"] = score

    pair_s1 = first_rows >> 32
    order = np.lexsort((-best_score, best_rank, pair_s1))       # by s1_row, then rank, then score
    sorted_s1 = pair_s1[order]
    group_start = np.searchsorted(sorted_s1, sorted_s1, side="left")
    candidate_rank = np.arange(len(order)) - group_start

    keep = order[candidate_rank < max_candidates]
    rescue_mask = np.zeros(n_pairs, dtype=bool)
    if rescue_n and len(keep):
        # the best `rescue` pairs per record: rank max_candidates .. max_candidates+rescue_n-1,
        # i.e. directly below the last kept candidate (candidate_rank is already per record)
        rank_of = np.empty(n_pairs, dtype=np.int64)
        rank_of[order] = candidate_rank
        rescue_mask = (rank_of >= max_candidates) & (rank_of < max_candidates + rescue_n)

    kept = candidate_rank[keep]
    common = {
        "s1_row": pair_s1[keep].astype(hits["s1_row"].dtype),
        "s23_row": (first_rows[keep] & ((1 << 32) - 1)).astype(hits["s23_row"].dtype),
        "best_rank": best_rank[keep], "best_score": best_score[keep], "n_methods": n_methods[keep],
        "candidate_rank": kept,
        **{name: values[keep] for name, values in method_scores.items()},
    }
    rescued = None
    if rescue_mask.any():
        rescued = pd.DataFrame({
            "s1_row": pair_s1[rescue_mask].astype(hits["s1_row"].dtype),
            "s23_row": (first_rows[rescue_mask] & ((1 << 32) - 1)).astype(hits["s23_row"].dtype),
            "best_rank": best_rank[rescue_mask], "best_score": best_score[rescue_mask],
            "n_methods": n_methods[rescue_mask], "candidate_rank": np.full(rescue_mask.sum(), -1, dtype=kept.dtype),
            **{name: values[rescue_mask] for name, values in method_scores.items()},
        })
    return pd.DataFrame(common), rescued


def combine_and_trim_in_pieces(hits, max_candidates, rescue=0, rows_per_piece=10_000_000):
    """combine_and_trim for a few Source 1 records at a time: all hits of one Source 1
    record are in the same piece, so the result is the same (also in the same order),
    but it needs far less memory (101M hits of the US train data at once went over 30 GB).
    Rescue rows (candidate_rank = -1) get their `search_margin` feature here: the gap to the
    best kept candidate of their record (all of which are in the same piece)."""
    if len(hits) <= rows_per_piece:
        candidates, rescued = combine_and_trim(hits, max_candidates, rescue)
        parts = [candidates] if rescued is None else [candidates, rescued]
    else:
        hits["method"] = hits["method"].astype("category")
        s1_row = hits["s1_row"].to_numpy()
        n_pieces = -(-len(hits) // rows_per_piece)
        edges = np.quantile(s1_row, np.linspace(0, 1, n_pieces + 1)[1:-1]).astype(np.int64)
        piece = np.searchsorted(np.unique(edges), s1_row, side="right")
        order = np.argsort(piece, kind="stable")
        starts = np.searchsorted(piece[order], np.arange(piece.max() + 2))
        parts = []
        for i in range(len(starts) - 1):
            rows = order[starts[i]:starts[i + 1]]
            if len(rows):
                candidates, rescued = combine_and_trim(hits.iloc[rows], max_candidates, rescue)
                parts.append(candidates)
                if rescued is not None:
                    parts.append(rescued)
                log(f"    combined piece {i + 1}/{len(starts) - 1}")
        del order, piece
    if not parts:
        return pd.DataFrame(columns=["s1_row", "s23_row", "candidate_rank"])
    table = pd.concat(parts, ignore_index=True)
    # search_margin: for rescue rows, the gap down from the record's best kept candidate.
    # NaN for candidates (they were kept by the search itself); the model reads the number as
    # "this pair was NOT close enough to be kept, how far below was it?".
    is_rescue = table["candidate_rank"].to_numpy() < 0
    table["search_margin"] = np.nan
    if is_rescue.any():
        scores = table["best_score"].to_numpy()
        best_kept = pd.Series(np.where(~is_rescue, scores, -np.inf)).groupby(table["s1_row"].to_numpy()).max()
        margin = best_kept.reindex(table["s1_row"].to_numpy()).to_numpy() - scores
        table.loc[is_rescue, "search_margin"] = margin[is_rescue]
    return table


# ---------------------------------------------------------------------------
# All countries
# ---------------------------------------------------------------------------
def build_candidates(s1, s23, config, query_mask=None, dry_run=False, checkpoint_dir=None, countries=None):
    """Candidates for every Source 1 record where query_mask is True
    (default: all of them).

    s1, s23 : cleaned tables from normalize.py (with a 0..n-1 index)
    dry_run : only print the amount of work of every search (to estimate time)
    checkpoint_dir : the finished candidates of each country (candidates_part_<country>.parquet)
              and every finished search (search_<country>_<retriever>_fwd2/fwd3/rev.parquet)
              are saved here. After a crash or time-out the run skips what is saved.
    countries : only these country labels (None = all). Used to spread the blocking of one
              split over several notebooks that run at the same time.
    Returns one row per candidate pair with the columns
        s1_row, s23_row, candidate_rank, best_rank, best_score, n_methods, score_<method>...
    """
    if query_mask is None:
        query_mask = np.ones(len(s1), dtype=bool)

    results = []
    for country, s1_group in s1.groupby("country", sort=True):
        if countries is not None and str(country) not in countries:
            continue
        s23_group = s23[s23["country"] == country]
        is_query = query_mask[s1_group.index.to_numpy()]
        log(f"country '{country}': {is_query.sum():,} queries, "
            f"{len(s1_group):,} Source 1, {len(s23_group):,} Source 2/3")
        if len(s23_group) == 0 or not is_query.any():
            continue

        safe_name = "".join(ch if ch.isalnum() else "_" for ch in str(country))
        part_file = None
        if checkpoint_dir is not None and not dry_run:
            part_file = Path(checkpoint_dir) / f"candidates_part_{safe_name}.parquet"
            if part_file.exists():
                # a part saved with a larger max_candidates: its candidate_rank is the same
                # ranking, so keeping the first max_candidates = trimming to max_candidates
                # (the rescue rows, candidate_rank = -1, always stay)
                part = pd.read_parquet(part_file)
                if "search_margin" not in part.columns:      # a part of an older run
                    part["search_margin"] = np.nan
                results.append(part[part["candidate_rank"] < config.max_candidates].reset_index(drop=True))
                log(f"  reusing the saved candidates of '{country}'")
                continue

        hits = [run_retriever(s1_group, s23_group, retriever, config, is_query, dry_run,
                              None if checkpoint_dir is None else
                              Path(checkpoint_dir) / f"search_{safe_name}_{retriever.name}")
                for retriever in config.retrievers]
        hits = pd.concat(hits, ignore_index=True)
        log(f"  {len(hits):,} raw hits")
        if len(hits):
            part = combine_and_trim_in_pieces(hits, config.max_candidates, config.rescue)
            results.append(part)
            if part_file is not None:
                part_file.parent.mkdir(parents=True, exist_ok=True)
                part.to_parquet(part_file)
                log(f"  saved {part_file.name}")
                # the saved searches of this country are not needed any more (saves GBs of output)
                for search_file in part_file.parent.glob(f"search_{safe_name}_*.parquet"):
                    search_file.unlink()

    if not results:
        return pd.DataFrame(columns=["s1_row", "s23_row"])
    candidates = pd.concat(results, ignore_index=True)
    candidates[["s1_row", "s23_row"]] = candidates[["s1_row", "s23_row"]].astype(np.int32)
    score_columns = [c for c in candidates.columns if c.startswith(("score_", "best_score"))]
    if "search_margin" in candidates.columns:
        score_columns.append("search_margin")
    candidates[score_columns] = candidates[score_columns].astype(np.float32)
    return candidates

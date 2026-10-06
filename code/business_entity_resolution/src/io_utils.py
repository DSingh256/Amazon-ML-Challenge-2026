"""
S0 - Reading and writing files.

All challenge files are tab-separated. Business names can contain quote
characters, so we turn quote handling OFF (QUOTE_NONE) and read every value
as plain text. Empty cells stay "" (never NaN).
"""
import csv
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
_START_TIME = time.time()


def memory_gb() -> float:
    """Memory used by this program (GB), or NaN when psutil is not installed."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return float("nan")


def log(message: str) -> None:
    """Print a message with the minutes passed since the program started and the
    memory in use (to see how close a step comes to the machine's limit)."""
    minutes = (time.time() - _START_TIME) / 60
    print(f"[{minutes:6.1f} min {memory_gb():5.1f} GB] {message}", flush=True)


def read_tsv(path) -> pd.DataFrame:
    """Read a challenge .tsv file. Every column is text, empty cells are ""."""
    return pd.read_csv(
        path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )


def source_file(data_dir, split: str, source: int) -> Path:
    """Path of e.g. <data_dir>/train/train_source2.tsv"""
    return Path(data_dir) / split / f"{split}_source{source}.tsv"


def read_source1(data_dir, split: str) -> pd.DataFrame:
    """Source 1 records (the reference list we must answer for)."""
    return read_tsv(source_file(data_dir, split, 1))


def read_source23(data_dir, split: str) -> pd.DataFrame:
    """Source 2 and Source 3 records in one table, with a 'source' column (2 or 3)."""
    parts = []
    for source in (2, 3):
        part = read_tsv(source_file(data_dir, split, source))
        part["source"] = source
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def read_ground_truth_pairs(data_dir) -> pd.DataFrame:
    """Training answers as one row per true link: (s1_id, s23_id).

    Source 1 records with no matches do not appear in this table.
    """
    truth = read_tsv(Path(data_dir) / "train" / "train_ground_truth.tsv")
    truth["s23_id"] = truth["matched_entity_ids"].str.split(",")
    pairs = truth.explode("s23_id")
    pairs = pairs[pairs["s23_id"] != ""]
    pairs = pairs.rename(columns={"source1_entity_id": "s1_id"})
    return pairs[["s1_id", "s23_id"]].reset_index(drop=True)


def truth_as_rows(true_pairs, s1, s23) -> pd.DataFrame:
    """Turn ground-truth ids into row numbers of the cleaned tables:
    columns s1_row, s23_row. Pairs whose ids are not in the tables are dropped
    (happens on samples)."""
    s1_row = pd.Index(s1["entity_id"]).get_indexer(true_pairs["s1_id"])
    s23_row = pd.Index(s23["entity_id"]).get_indexer(true_pairs["s23_id"])
    known = (s1_row >= 0) & (s23_row >= 0)
    return pd.DataFrame({"s1_row": s1_row[known], "s23_row": s23_row[known]})


def id_bucket(ids) -> np.ndarray:
    """A number 0..999 for every id, always the same on every computer.
    Used to pick fixed random parts of the data (holdout, training sample)."""
    return np.array([zlib.crc32(i.encode()) % 1000 for i in ids])


def write_id_lists(path, s1_ids, pairs: pd.DataFrame, s23_ids, header: str) -> None:
    """Write the submission format: one row per Source 1 record, then a comma list.

    s1_ids  : every Source 1 id, in row order (= the original file order)
    pairs   : table with the row numbers 's1_row' and 's23_row'
    s23_ids : Source 2/3 ids in row order (to turn s23_row back into an id)
    header  : name of the second column, e.g. "matched_entity_ids"
    Source 1 records without any pair get an empty list.
    """
    pairs = pairs.drop_duplicates(["s1_row", "s23_row"])
    matched = pd.DataFrame({"s1_row": pairs["s1_row"].to_numpy(),
                            "s23_id": np.asarray(s23_ids, dtype=object)[pairs["s23_row"].to_numpy()]})
    joined = matched.groupby("s1_row")["s23_id"].agg(",".join)

    table = pd.DataFrame({"source1_entity_id": list(s1_ids)})
    table[header] = joined.reindex(np.arange(len(table))).fillna("").to_numpy()

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False, quoting=csv.QUOTE_NONE, lineterminator="\n")

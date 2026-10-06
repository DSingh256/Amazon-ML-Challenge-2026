"""
Make a small copy of the TRAIN data that fits on a laptop.

What goes into the sample:
  1. N random Source 1 records
  2. all their true matches from Source 2 and Source 3
  3. extra random Source 2/3 records that belong to nobody in the sample
     ("distractors"), so the search is not too easy

The sample is written in the same folder layout as the real data
(<out_dir>/train/train_source1.tsv ...), so every other script can run on
it without any change.

With --cities the sample is a DENSE world instead: every Source 1 and Source 2/3 record
whose address contains one of the given city words (plus the true matches of those
Source 1 records). All look-alike businesses of a city are then present, as in the
full data - the random sample above is much easier than the real task. (Only for
testing on a laptop; the pipeline itself never looks at city names.)

The big files are read line by line, so this never needs much memory.

Usage:
    python src/make_sample.py --data-dir D:/AmazonML/student_resource/dataset \
                              --out-dir  D:/AmazonML/work/sample_data --n-s1 30000
"""
import argparse
import csv
import random
import re
from pathlib import Path

from config import SEED

csv.field_size_limit(2**31 - 1)   # allow very long fields

# The full training set has about 4.7 Source 2/3 records per Source 1 record.
# We keep the same ratio in the sample.
S23_PER_S1_IN_FULL_DATA = 4.7
TOTAL_S23_IN_FULL_TRAIN = 10_320_219


def read_rows(path):
    """Yield (header, row) one line at a time."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        for row in reader:
            yield header, row


def write_rows(path, header, rows):
    """Write rows back exactly as they were read (tab between values)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        for row in [header] + rows:
            f.write("\t".join(row) + "\n")


def dense_sample(train_in, train_out, cities):
    """Every record whose address contains one of the city words, plus true matches."""
    pattern = re.compile(r"\b(" + "|".join(re.escape(c.strip()) for c in cities) + r")\b", re.IGNORECASE)
    s1_rows = []
    for header, row in read_rows(train_in / "train_source1.tsv"):
        s1_header = header
        if pattern.search(row[2]):
            s1_rows.append(row)
    chosen_s1 = {row[0] for row in s1_rows}
    write_rows(train_out / "train_source1.tsv", s1_header, s1_rows)
    print(f"Source 1: kept {len(s1_rows):,} records")
    truth_rows, needed_s23 = [], set()
    for header, row in read_rows(train_in / "train_ground_truth.tsv"):
        truth_header = header
        if row[0] in chosen_s1:
            truth_rows.append(row)
            needed_s23.update(x for x in row[1].split(",") if x)
    write_rows(train_out / "train_ground_truth.tsv", truth_header, truth_rows)
    for source in (2, 3):
        kept, n_extra = [], 0
        for header, row in read_rows(train_in / f"train_source{source}.tsv"):
            if row[0] in needed_s23:
                kept.append(row)
            elif pattern.search(row[2]):
                kept.append(row)
                n_extra += 1
        write_rows(train_out / f"train_source{source}.tsv", header, kept)
        print(f"Source {source}: kept {len(kept):,} records ({n_extra:,} same-city records of nobody here)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="folder that contains train/ and test/")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-s1", type=int, default=30_000)
    parser.add_argument("--cities", default="", help="comma-separated city words: dense sample (see above)")
    args = parser.parse_args()
    if args.cities:
        dense_sample(Path(args.data_dir) / "train", Path(args.out_dir) / "train", args.cities.split(","))
        return

    rng = random.Random(SEED)
    train_in = Path(args.data_dir) / "train"
    train_out = Path(args.out_dir) / "train"

    # 1. Pick random Source 1 ids (reservoir sampling = one pass, fixed memory).
    s1_header, s1_rows = None, []
    for i, (header, row) in enumerate(read_rows(train_in / "train_source1.tsv")):
        s1_header = header
        if len(s1_rows) < args.n_s1:
            s1_rows.append(row)
        else:
            j = rng.randint(0, i)
            if j < args.n_s1:
                s1_rows[j] = row
    chosen_s1 = {row[0] for row in s1_rows}
    write_rows(train_out / "train_source1.tsv", s1_header, s1_rows)
    print(f"Source 1: kept {len(s1_rows):,} records")

    # 2. Ground truth rows of the chosen Source 1 ids -> the Source 2/3 ids we need.
    truth_rows, needed_s23 = [], set()
    for header, row in read_rows(train_in / "train_ground_truth.tsv"):
        truth_header = header
        if row[0] in chosen_s1:
            truth_rows.append(row)
            needed_s23.update(x for x in row[1].split(",") if x)
    write_rows(train_out / "train_ground_truth.tsv", truth_header, truth_rows)
    print(f"Ground truth: {len(truth_rows):,} rows, {len(needed_s23):,} matched Source 2/3 ids")

    # 3. Source 2/3: keep the true matches plus random distractors.
    target_total = int(args.n_s1 * S23_PER_S1_IN_FULL_DATA)
    extra_needed = max(target_total - len(needed_s23), 0)
    keep_probability = extra_needed / TOTAL_S23_IN_FULL_TRAIN

    for source in (2, 3):
        kept, n_extra = [], 0
        name = f"train_source{source}.tsv"
        for header, row in read_rows(train_in / name):
            if row[0] in needed_s23:
                kept.append(row)
            elif rng.random() < keep_probability:
                kept.append(row)
                n_extra += 1
        write_rows(train_out / name, header, kept)
        print(f"Source {source}: kept {len(kept):,} records ({n_extra:,} distractors)")


if __name__ == "__main__":
    main()

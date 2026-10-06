"""Apply dec1's TOP1 decoder (keep p >= 0.74, or the best candidate of a record if p >= 0.56) to a notebook's
test_pair_probabilities.parquet and write matching_results.tsv (same writer as ctx1.py)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc

src, out = Path(sys.argv[1]), Path(sys.argv[2])
T, T1 = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.74, 0.56)
RAW = Path("D:/AmazonML/student_resource/dataset/test")
t = pd.read_parquet(src, columns=["s1_row", "s23_row", "probability"])
s1, s23, p = (t[c].to_numpy() for c in ("s1_row", "s23_row", "probability"))
del t
o = np.lexsort((-p, s23))                      # exclusivity: best Source 1 record per Source 2/3 record
first = np.ones(len(o), dtype=bool)
first[1:] = s23[o][1:] != s23[o][:-1]
k = o[first]
k = k[np.lexsort((-p[k], s1[k]))]
s1, s23, p = s1[k], s23[k], p[k]
start = np.r_[True, s1[1:] != s1[:-1]]
keep = (p >= T) | (start & (p >= T1))


def ids(name):
    tb = pc.read_csv(RAW / name, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                     convert_options=pc.ConvertOptions(include_columns=["entity_id"],
                                                       column_types={"entity_id": pa.string()}))
    return tb.column("entity_id").combine_chunks()


s1_ids = ids("test_source1.tsv").to_numpy(zero_copy_only=False)
s23_ids = pa.concat_arrays([ids("test_source2.tsv"), ids("test_source3.tsv")])
a, b = s1[keep], s23[keep]
sid = np.asarray(s23_ids.take(pa.array(b)).to_numpy(zero_copy_only=False), dtype=object)
joined = pd.Series(sid).groupby(a).agg(",".join)
col = joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()
out.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": col}).to_csv(out, sep="\t", index=False, lineterminator="\n")
print(f"TOP1({T},{T1}): {keep.sum():,} pairs, {len(joined):,} records answered -> {out}")

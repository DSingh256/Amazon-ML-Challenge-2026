"""
avg_probs: average the re-scored pair probabilities of two pipelines (a pair missing in one counts as
0 there), then exclusivity + the keep rule (t, t1). Less jumpy than switching pipelines outright.
Usage: python ctx/avg_probs.py <probs_a.parquet> <probs_b.parquet> <weight_b> <t> <t1|None> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc

sys.path.insert(0, str(Path(__file__).parent))
from ctx1 import exclusive_sorted, keep_rule  # noqa: E402
import ctx4  # noqa: E402

pa_, pb_, wb, t, t1, out_dir = sys.argv[1:7]
wb, t = float(wb), float(t)
t1 = None if t1 == "None" else float(t1)
out_dir = Path(out_dir)
cols = ["s1_row", "s23_row", "probability"]
A = pd.read_parquet(pa_, columns=cols)
B = pd.read_parquet(pb_, columns=cols)
m = A.merge(B, on=["s1_row", "s23_row"], how="outer", suffixes=("_a", "_b"))
del A, B
q = ((1 - wb) * m["probability_a"].fillna(0).to_numpy(np.float64)
     + wb * m["probability_b"].fillna(0).to_numpy(np.float64))
s1, s23 = m["s1_row"].to_numpy(), m["s23_row"].to_numpy()
print(f"{len(m):,} pairs; only in a {m['probability_b'].isna().sum():,}, only in b {m['probability_a'].isna().sum():,}")
del m
o = exclusive_sorted(s1, s23, q)
s1, s23, q = s1[o], s23[o], q[o]
o = np.lexsort((-q, s1))
s1, s23, q = s1[o], s23[o], q[o]
rank, _, _ = ctx4.rank_in_group(s1)
keep = keep_rule(q, rank, t, t1)
raw = Path("D:/AmazonML/student_resource/dataset/test")


def ids(name):
    tb = pc.read_csv(raw / name, parse_options=pc.ParseOptions(delimiter="\t", quote_char=False),
                     convert_options=pc.ConvertOptions(include_columns=["entity_id"],
                                                       column_types={"entity_id": pa.string()}))
    return tb.column("entity_id").combine_chunks()


s1_ids = ids("test_source1.tsv").to_numpy(zero_copy_only=False)
s23_ids = pa.concat_arrays([ids("test_source2.tsv"), ids("test_source3.tsv")])
a, b = s1[keep], s23[keep]
sid = np.asarray(s23_ids.take(pa.array(b)).to_numpy(zero_copy_only=False), dtype=object)
joined = pd.Series(sid).groupby(a).agg(",".join)
col = joined.reindex(np.arange(len(s1_ids))).fillna("").to_numpy()
out_dir.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"source1_entity_id": s1_ids, "matched_entity_ids": col}).to_csv(
    out_dir / "matching_results.tsv", sep="\t", index=False, lineterminator="\n")
pd.DataFrame({"s1_row": s1, "s23_row": s23, "probability": q.astype(np.float32)}).to_parquet(
    out_dir / "test_pair_probabilities_avg.parquet", index=False)
print(f"wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs, {len(joined):,} records answered")

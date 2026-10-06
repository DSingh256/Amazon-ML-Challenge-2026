"""
apply_rule: exclusivity + keep rule (q >= t, or rank-0 q >= t1) on one pipeline's test probabilities.
Pairs below 0.3 are dropped first (cannot change an exclusivity winner that is kept, nor a kept rank 0).
Usage: python ctx/apply_rule.py <probs.parquet> <t> <t1|None> <out_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from ctx1 import exclusive_sorted, keep_rule  # noqa: E402
import ctx4  # noqa: E402

probs, t, t1, out_dir = sys.argv[1], float(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
t1 = None if t1 == "None" else float(t1)
d = pq.read_table(probs, columns=["s1_row", "s23_row", "probability"], filters=[("probability", ">=", 0.3)]).to_pandas()
s1, s23, q = d["s1_row"].to_numpy(), d["s23_row"].to_numpy(), d["probability"].to_numpy().astype(np.float64)
del d
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
print(f"wrote {out_dir / 'matching_results.tsv'}: {keep.sum():,} pairs, {len(joined):,} records answered (rule {t}, {t1})")

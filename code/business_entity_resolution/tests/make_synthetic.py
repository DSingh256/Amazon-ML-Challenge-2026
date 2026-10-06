"""
Tiny synthetic dataset for an end-to-end smoke test of run_pipeline.py, no challenge data needed.

Builds many small businesses, each with 1-3 noisy copies in Source 2 / Source 3, plus decoy
records that belong to nobody. Splits by business name: train and test share no business.

    <out>/train/train_source{1,2,3}.tsv  +  train_ground_truth.tsv
    <out>/test/test_source{1,2,3}.tsv

Run:  py -3.12 tests/make_synthetic.py --out-dir work/smoke_data
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CITIES = ["Springfield", "Riverside", "Fairview", "Georgetown"]
STREETS = ["Main Street", "Oak Avenue", "Market Road", "Station Road"]
COUNTRIES = ["India", "US"]
SUFFIX = ["Ltd", "Inc", "LLC", "Corp", "LLP", "Pvt Ltd"]


def noisy_copy(rng, name, address):
    words = name.split()
    kind = rng.integers(0, 4)
    if kind == 0:
        words = words + [str(rng.choice(["Ltd", "Private Limited", "Incorporated"]))]
    elif kind == 1 and len(words) > 2:
        words[0] = words[0][::-1]                     # scrambled first word
    elif kind == 2:
        words = list(dict.fromkeys(words + words[:1]))  # repeated word
    copy = " ".join(words)
    if rng.random() < 0.2:
        copy = copy.lower()
    if rng.random() < 0.15:
        address = address.split(",")[0]               # street only, city dropped
    return copy, address


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-s1", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    firsts = [a + b for a in "ABCDEFGH" for b in "AEIOUBCDFGKLMNPSTRVZ"]
    middles = ["Traders", "Logistics", "Foods", "Enterprises", "Services", "Retail", "Textiles", "Hardware"]
    names = [f"{f} {m} {s}" for f in firsts for m in middles for s in SUFFIX]
    rng.shuffle(names)
    names = names[:args.n_s1 * 3]

    s1, s2, s3, truth = [], [], [], []
    n23 = 0
    for i in range(args.n_s1):
        name = names[i]
        base = " ".join(name.split()[:2])             # the business identity (first two words)
        country = COUNTRIES[i % 2]
        address = f"{int(rng.integers(1, 500))} {rng.choice(STREETS)}, {rng.choice(CITIES)}"
        s1_id = f"S1_{i:06d}"
        s1.append([s1_id, name, address, country])
        for _ in range(int(rng.integers(1, 4))):
            n23 += 1
            copy, copy_addr = noisy_copy(rng, name, address)
            source = 2 if rng.random() < 0.6 else 3
            s23_id = f"S{source}_{n23:06d}"
            (s2 if source == 2 else s3).append([s23_id, copy, copy_addr, country])
            truth.append([s1_id, s23_id])

    for j in range(200):                              # decoys: belong to nobody
        n23 += 1
        name = f"{rng.choice(firsts)} {rng.choice(middles)} {rng.choice(SUFFIX)}"
        s3.append([f"S3_{n23:06d}", name, f"{int(rng.integers(1, 500))} {rng.choice(STREETS)}, {rng.choice(CITIES)}",
                   COUNTRIES[int(rng.integers(0, 2))]])

    s1 = pd.DataFrame(s1, columns=["entity_id", "business_name", "business_address", "country"])
    s2 = pd.DataFrame(s2, columns=["entity_id", "business_name", "business_address", "country"])
    s3 = pd.DataFrame(s3, columns=["entity_id", "business_name", "business_address", "country"])
    truth = pd.DataFrame(truth, columns=["source1_entity_id", "matched_entity_ids"])

    owner = dict(zip(s1["entity_id"], s1["business_name"].str.split().str[:2].str.join(" ")))
    truth["base"] = truth["source1_entity_id"].map(owner)
    bases = sorted(truth["base"].unique())
    train_bases = set(bases[:len(bases) // 2])

    out = Path(args.out_dir)
    for split, keep in (("train", truth["base"].isin(train_bases)), ("test", ~truth["base"].isin(train_bases))):
        ids = truth.loc[keep, "source1_entity_id"].unique()
        part1 = s1[s1["entity_id"].isin(ids)].reset_index(drop=True)
        s23_ids = set(truth.loc[keep, "matched_entity_ids"])
        part2 = s2[s2["entity_id"].isin(s23_ids)].reset_index(drop=True)
        part3 = s3[s3["entity_id"].isin(s23_ids) | ~s3["entity_id"].str.startswith("S3")].reset_index(drop=True)
        # decoys: put half in train, half in test
        if split == "test":
            part3 = pd.concat([part3, s3[~s3["entity_id"].str.startswith("S3")]], ignore_index=True)
        (out / split).mkdir(parents=True, exist_ok=True)
        part1.to_csv(out / split / f"{split}_source1.tsv", sep="\t", index=False)
        part2.to_csv(out / split / f"{split}_source2.tsv", sep="\t", index=False)
        part3.to_csv(out / split / f"{split}_source3.tsv", sep="\t", index=False)
        if split == "train":
            truth_out = truth.loc[keep, ["source1_entity_id", "matched_entity_ids"]]
            truth_out.to_csv(out / split / "train_ground_truth.tsv", sep="\t", index=False)
    print(f"synthetic data in {out}: {len(s1)} Source 1, {len(s2) + len(s3)} Source 2/3, "
          f"{len(bases) // 2} train businesses / {len(bases) - len(bases) // 2} test businesses")


if __name__ == "__main__":
    main()

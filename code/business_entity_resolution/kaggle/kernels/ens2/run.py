"""
aml26-ens2: the FINAL combination, CPU only. v27.
    base      : aml26-wide30 (30 candidates per record instead of 15: more true matches reachable)
    LLM layer : mean cross-encoder score of EVERY aml26-llm<N> notebook in the inputs (llm2..llm7)
    self-training: sure test pairs of wide30's model become extra stage-2 learning pairs (as pseudo1)
(below: the pseudo1 description, which still holds)

aml26-pseudo1: self-training (pseudo-labels) on the TEST set, CPU only. v25.

The test set has a country the training data does not have (France). The model never saw
a French match, so its French probabilities are less right. Self-training lets the model
see test data: the test pairs the current model (SRC = aml26-ens1) is almost sure about
become extra training examples for stage 2.
    positive : the pair keeps its Source 2/3 record after exclusivity and p >= POS
    negative : p <= NEG
No hand labels, no external data, no country rule: every test record is treated the same.

Steps
    gen    : SRC's models score all test pairs (stage 1 -> collective -> cross-encoder -> stage 2,
             exactly like assign.py), pick the sure pairs, write work/pseudo/pseudo_pairs.parquet
             with every stage-2 feature + label
    train  : train.py patched: the pseudo pairs are added to the LEARNING part only (early
             stopping and the holdout stay real train labels, so the holdout still tells
             whether India / US got worse)
    assign : the new model -> matching_results.tsv (+ t0.5/0.6/0.8 files, test probabilities)
"""
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

NOTEBOOK_START = time.time()
SRC = "aml26-wide30"
NOTE = "v27 ens2: wide30 (30 candidates) + mean cross-encoder of all llm notebooks + pseudo-labels"
POS, NEG = 0.95, 0.02
PSEUDO_SHARE = 0.30        # pseudo pairs at most 30% of the real learning pairs
EXTRA_THRESHOLDS = (0.5, 0.6, 0.8)
N_JOBS = 4
PACKAGES = ["anyascii==0.3.3", "rapidfuzz==3.14.6", "sparse_dot_topn==1.2.0"]
PIPELINE_TIME_LIMIT = 11.25 * 3600

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"
WORK = WORKING / "work"
SRC_MODELS = WORKING / "src_models"


def say(message):
    print(f"### {message}", flush=True)


def find(name):
    return sorted(p for root in (INPUT, WORKING / "unzipped") if root.exists() for p in root.rglob(name))


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.symlink(source, target)


def combine_ce_scores():
    """The ensemble: for every pair, the mean of the cross-encoder scores (logits) of all
    LLM-layer notebooks given as input (aml26-llm<N>). Written to WORK/<split>/ce_scores.parquet
    BEFORE the other inputs are linked, so train / assign use the mean instead of one model."""
    import re
    import pandas as pd
    for split in ("train", "test"):
        files = sorted(p for p in INPUT.rglob(f"work/{split}/ce_scores.parquet")
                       if re.search(r"aml26-llm\d", str(p)))
        if not files:
            raise SystemExit(f"no {split} ce_scores.parquet of an aml26-llm notebook in the inputs")
        parts = []
        for i, path in enumerate(files):
            part = pd.read_parquet(path, columns=["s1_row", "s23_row", "ce"])
            say(f"{split}: {path} -> {len(part):,} pairs, ce mean {part['ce'].mean():.3f} sd {part['ce'].std():.3f}")
            parts.append(part.rename(columns={"ce": f"ce{i}"}).set_index(["s1_row", "s23_row"]))
        wide = pd.concat(parts, axis=1, join="outer")
        common = wide.dropna()
        say(f"{split}: {len(common):,} pairs scored by all {len(files)} models; correlations:\n{common.corr().round(4)}")
        out = wide.mean(axis=1).rename("ce").astype("float32").reset_index()
        target = WORK / split / "ce_scores.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
        say(f"{split}: mean score of {len(out):,} pairs -> {target}")


def prepare():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    if not find("test_source1.tsv"):
        for archive in find("*.zip"):
            zipfile.ZipFile(archive).extractall(WORKING / "unzipped")
    for split, names in {"train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                                   "train_ground_truth.tsv"],
                         "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]}.items():
        for name in names:
            link(find(name)[0], DATA / split / name)
    # SRC's cross-encoder scores (what its model was trained with): gen uses them to reproduce SRC's p
    found = [p for p in INPUT.rglob("work/test/ce_scores.parquet") if SRC in str(p)]
    if not found:
        raise SystemExit(f"{SRC} work/test/ce_scores.parquet not found (did {SRC} finish?)")
    link(found[0], WORK / "test" / "ce_src.parquet")
    say(f"test/ce_src.parquet <- {found[0]}")
    combine_ce_scores()      # the new mean of all llm notebooks -> work/<split>/ce_scores.parquet
    meta = [p for p in INPUT.rglob("work/models/meta.json") if SRC in str(p)]
    if not meta:
        raise SystemExit(f"{SRC} models not found")
    for path in meta[0].parent.iterdir():
        link(path, SRC_MODELS / path.name)
    # everything else (features, cleaned tables, candidates): SRC's files first (its 30-candidate
    # features win), then the block notebooks; the llm notebooks' scores are already in the mean
    import re
    count = 0
    paths = [p for p in INPUT.rglob("work/*/*") if p.is_file() and p.parent.name in ("train", "test")
             and not re.search(r"aml26-llm\d", str(p))]
    for path in sorted(paths, key=lambda p: SRC not in str(p)):
        if not (WORK / path.parent.name / path.name).exists():
            link(path, WORK / path.parent.name / path.name)
            count += 1
    say(f"reusing {count} files of the input notebooks")
    for needed in ["train/features.parquet", "train/candidates.parquet", "test/features.parquet",
                   "test/candidate_pairs.tsv", "test/s1_clean.parquet", "test/s23_clean.parquet"]:
        if not (WORK / needed).exists():
            raise SystemExit(f"work/{needed} missing")


def code_dir():
    return find("run_pipeline.py")[0].parent


# ---------------------------------------------------------------------------
# gen: the pseudo-labelled test pairs (runs in its own process: frees its memory)
# ---------------------------------------------------------------------------
def gen():
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import pyarrow.parquet as pq
    sys.path.insert(0, str(code_dir()))
    from stage2 import collective_features
    from cross_encoder import ce_features

    split_dir = WORK / "test"
    meta = json.loads((SRC_MODELS / "meta.json").read_text())
    model = lgb.Booster(model_file=str(SRC_MODELS / "lgbm.txt"))
    stage1 = [lgb.Booster(model_file=str(SRC_MODELS / n)) for n in meta["stage1_models"]]
    first = meta["stage1_features"]
    file = pq.ParquetFile(split_dir / "features.parquet")

    def batches(columns):
        for batch in file.iter_batches(batch_size=1_000_000, columns=["s1_row", "s23_row"] + columns):
            yield batch.to_pandas()

    rows, p1 = [], []
    for batch in batches(first):
        rows.append(batch[["s1_row", "s23_row"]].to_numpy())
        p1.append((sum(m.predict(batch[first]) for m in stage1) / len(stage1)).astype(np.float32))
    rows = np.concatenate(rows)
    p1 = np.concatenate(p1)
    s1_row, s23_row = rows[:, 0], rows[:, 1]
    del rows
    say(f"gen: stage 1 of {len(p1):,} test pairs done")
    s23 = pd.read_parquet(split_dir / "s23_clean.parquet", columns=["name_core", "addr_clean"])
    extra = collective_features([(s1_row, s23_row, p1)], s23["name_core"].to_numpy(),
                                s23["addr_clean"].to_numpy())[0]
    del s23
    collective = extra
    scores = pd.read_parquet(split_dir / "ce_src.parquet")
    extra = pd.concat([collective, ce_features(s1_row, s23_row, scores)], axis=1)
    del scores
    say(f"gen: collective + cross-encoder features ready {list(extra.columns)}")

    p = np.empty(len(p1), dtype=np.float32)
    start = 0
    for batch in batches(first):
        stop = start + len(batch)
        batch = pd.concat([batch.reset_index(drop=True), extra.iloc[start:stop].reset_index(drop=True)], axis=1)
        p[start:stop] = model.predict(batch[meta["features"]])
        start = stop
    saved = [q for q in INPUT.rglob("test_pair_probabilities.parquet") if SRC in str(q)]
    if saved:
        old = pd.read_parquet(saved[0])["probability"].to_numpy()
        if len(old) == len(p):
            say(f"gen: check vs {SRC}'s saved probabilities: max abs diff {np.abs(old - p).max():.2e}")

    del extra
    gc.collect()
    scores = pd.read_parquet(split_dir / "ce_scores.parquet")
    new_ce = ce_features(s1_row, s23_row, scores)
    del scores
    say(f"gen: new cross-encoder mean: {new_ce.notna().mean().round(3).to_dict()} of pairs scored")
    extra = pd.concat([collective, new_ce], axis=1)
    del collective, new_ce

    # exclusivity winners: the best s1 for every s23
    order = np.lexsort((-p, s23_row))
    winner = np.zeros(len(p), dtype=bool)
    first_of = np.r_[True, s23_row[order][1:] != s23_row[order][:-1]]
    winner[order[first_of]] = True
    positive = winner & (p >= POS)
    negative = p <= NEG

    labels = pq.read_table(WORK / "train" / "features.parquet", columns=["label"]).column(0).to_numpy()
    n_learn = int(len(labels) * 0.8)
    rate = float(labels.mean())
    del labels
    cap_pos, cap_neg = int(PSEUDO_SHARE * n_learn * rate), int(PSEUDO_SHARE * n_learn * (1 - rate))
    rng = np.random.default_rng(42)

    def sample(mask, cap):
        idx = np.flatnonzero(mask)
        return np.sort(rng.choice(idx, cap, replace=False)) if len(idx) > cap else idx

    pos_idx, neg_idx = sample(positive, cap_pos), sample(negative, cap_neg)
    say(f"gen: sure test pairs: {positive.sum():,} positive (p>={POS}, exclusive), {negative.sum():,} negative "
        f"(p<={NEG}); train positive rate {rate:.3f}, learning pairs ~{n_learn:,}; "
        f"taking {len(pos_idx):,} + {len(neg_idx):,}")
    country = pd.read_parquet(split_dir / "s1_clean.parquet", columns=["country"])["country"].to_numpy()
    say("gen: positives by country (log only) " + str(pd.Series(country[s1_row[pos_idx]]).value_counts().to_dict()))
    say("gen: test records by country (log only) " + str(pd.Series(country).value_counts().to_dict()))
    say("gen: p between 0.3 and 0.9 (unsure) by country (log only) "
        + str(pd.Series(country[s1_row[winner & (p > 0.3) & (p < 0.9)]]).value_counts().to_dict()))

    take = np.zeros(len(p), dtype=bool)
    take[pos_idx] = True
    take[neg_idx] = True
    label = positive.astype(np.int8)
    parts = []
    start = 0
    for batch in batches(first):
        stop = start + len(batch)
        sel = take[start:stop]
        if sel.any():
            part = pd.concat([batch.reset_index(drop=True), extra.iloc[start:stop].reset_index(drop=True)], axis=1)[sel]
            part = part[meta["features"]].copy()
            part["label"] = label[start:stop][sel]
            parts.append(part)
        start = stop
    out = pd.concat(parts, ignore_index=True)
    (WORK / "pseudo").mkdir(parents=True, exist_ok=True)
    out.to_parquet(WORK / "pseudo" / "pseudo_pairs.parquet", index=False)
    say(f"gen: wrote {len(out):,} pseudo pairs ({out['label'].mean():.3f} positive), {out.shape[1]} columns")


# ---------------------------------------------------------------------------
# patched copy of the code
# ---------------------------------------------------------------------------
TRAIN_OLD = "    model, curve = train_model(learning, stopping, columns)\n"
TRAIN_NEW = """    pseudo_file = work_dir / "pseudo" / "pseudo_pairs.parquet"
    if pseudo_file.exists():
        pseudo = pd.read_parquet(pseudo_file)
        missing = [c for c in columns if c not in pseudo.columns]
        if missing:
            raise SystemExit(f"pseudo pairs lack the columns {missing}")
        n_real = len(learning)
        learning = pd.concat([learning[columns + ["label"]], pseudo[columns + ["label"]]], ignore_index=True)
        log(f"PSEUDO: learning = {n_real:,} train pairs + {len(pseudo):,} sure test pairs "
            f"({pseudo['label'].mean():.3f} positive); early stop + holdout stay train only")
        del pseudo
        gc.collect()
""" + TRAIN_OLD

ASSIGN_OLD = """    log(f"wrote {output_dir / 'matching_results.tsv'} and candidate_pairs.tsv")\n"""
ASSIGN_NEW = ASSIGN_OLD + f"""    try:
        extra_dir = output_dir / "thresholds"
        extra_dir.mkdir(parents=True, exist_ok=True)
        pairs[["s1_row", "s23_row", "probability"]].to_parquet(extra_dir / "test_pair_probabilities.parquet", index=False)
        for t in {EXTRA_THRESHOLDS!r}:
            extra = decide(pairs, "threshold", threshold=t)
            write_id_lists(extra_dir / f"matching_results_t{{t}}.tsv", s1_ids, extra, s23_ids, header="matched_entity_ids")
            log(f"threshold {{t}}: {{len(extra):,}} pairs -> thresholds/matching_results_t{{t}}.tsv")
    except Exception as error:
        log(f"extra threshold files failed (main answer is fine): {{error!r}}")
"""


def patched_code():
    code = Path("/tmp/aml26_code")
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(code_dir(), code)
    for name, old, new in (("train.py", TRAIN_OLD, TRAIN_NEW), ("assign.py", ASSIGN_OLD, ASSIGN_NEW)):
        text = (code / name).read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise SystemExit(f"{name} patch failed (code dataset changed?)")
        (code / name).write_text(text.replace(old, new), encoding="utf-8")
    return code


def run(command, label):
    time_left = PIPELINE_TIME_LIMIT - (time.time() - NOTEBOOK_START)
    say(f"{label}: {' '.join(map(str, command))} (time left {time_left / 3600:.2f} h)")
    try:
        returncode = subprocess.run(command, timeout=time_left).returncode
    except subprocess.TimeoutExpired:
        returncode = -1
    import resource
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    say(f"peak memory {peak:.1f} GB; time {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
    if returncode != 0:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "FAILED.txt").write_text(f"{label}: exit code {returncode}\n")
        say(f"PIPELINE FAILED ({label}: exit code {returncode}{' = killed, out of memory?' if returncode == -9 else ''})")
        return False
    say(f"PIPELINE OK ({label})")
    return True


def pipeline(code, *args):
    return run([sys.executable, "-u", str(code / "run_pipeline.py"), *args, "--data-dir", str(DATA),
                "--work-dir", str(WORK), "--n-jobs", str(N_JOBS)], args[0])


def validate():
    output = WORK / "output"
    validator = find("validate_submission.py")
    if validator and (output / "matching_results.tsv").exists():
        subprocess.run([sys.executable, str(validator[0]), "-m", str(output / "matching_results.tsv"),
                        "-c", str(output / "candidate_pairs.tsv"), "-t", str(DATA / "test")])


def clean_up():
    for path in WORKING.rglob("*"):
        if path.is_symlink():
            path.unlink()
    (WORK / "test" / "ce_src.parquet").unlink(missing_ok=True)
    for folder in (DATA, WORKING / "unzipped", SRC_MODELS):
        shutil.rmtree(folder, ignore_errors=True)
    (WORK / "pseudo" / "pseudo_pairs.parquet").unlink(missing_ok=True)   # big; the log has its numbers


def gen_in_child():
    """gen in a forked process (its ~10 GB go back to the system when it ends)."""
    import multiprocessing
    say("gen: pseudo-labelled test pairs")
    child = multiprocessing.get_context("fork").Process(target=gen)
    child.start()
    child.join()
    if child.exitcode != 0:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "FAILED.txt").write_text(f"gen: exit code {child.exitcode}\n")
        say(f"PIPELINE FAILED (gen: exit code {child.exitcode}{' = killed, out of memory?' if child.exitcode == -9 else ''})")
        return False
    say(f"PIPELINE OK (gen); time {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
    return True


def main():
    say(f"aml26-ens2: {NOTE}")
    try:
        prepare()
        code = patched_code()
        if (gen_in_child() and pipeline(code, "train", "--note", NOTE) and pipeline(code, "assign")):
            validate()
            say("ALL DONE")
    finally:
        clean_up()


main()

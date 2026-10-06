"""
Kaggle notebook script (one file for every step; prepare_kaggle.py sets MODE).

    MODE = "dry-run"   : clean the test data and print how much blocking work it is
    MODE = "train"     : clean + blocking + features + model on the full train data
    MODE = "test-prep" : clean + blocking + features on the test data
    MODE = "block <split> <countries>" : clean + blocking of those countries only
                         (train / test-prep with these notebooks as input reuse the parts)
    MODE = "llm"       : the "LLM layer" on a GPU: scores of the unsure train / test pairs
    MODE = "assign"    : score the test pairs and write output/*.tsv
                         (inputs: the outputs of the train and test-prep notebooks)

It finds everything under /kaggle/input by file name, so it works whatever
folder names Kaggle gives the datasets. Results of earlier notebooks that are
added as inputs (a work/ folder) are reused, so finished steps are skipped.
"""
import os
import resource
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

NOTEBOOK_START = time.time()
MODE = "wide"       # v24: 30 candidates per record -> train -> test-prep -> assign (one CPU notebook)
NOTE = "v24 wide30: 30 candidates per record (was 15) + mean cross-encoder score of llm2/llm3/llm4"
# v24: keep 30 candidates per Source 1 record instead of 15. The saved searches / candidate parts
# of the block notebooks were made with 30, so no new search is needed (train US is combined
# again from its saved searches). Dense sample: pair recall 0.9848 -> 0.9893, records with all
# matches found 0.9531 -> 0.9663, best possible F0.5 0.9950 -> 0.9964.
CONFIG_PATCH = {"max_candidates: int = 15": "max_candidates: int = 30"}
EXTRA_THRESHOLDS = (0.5, 0.6, 0.8)
N_JOBS = 4
PACKAGES = ["anyascii==0.3.3", "rapidfuzz==3.14.6", "sparse_dot_topn==1.2.0"]

# Kaggle stops a notebook after 12 hours and then saves NOTHING. So the pipeline is
# stopped a bit earlier: the notebook can still finish and save the finished steps
# (every country / search is saved as it completes), and a resume notebook continues.
PIPELINE_TIME_LIMIT = 11.25 * 3600      # seconds after the notebook started

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"          # links to the challenge .tsv files
WORK = WORKING / "work"          # pipeline results (saved as notebook output)


def say(message):
    print(f"### {message}", flush=True)


def find(name):
    """All files called `name` under /kaggle/input (and unzipped data)."""
    return sorted(p for root in (INPUT, WORKING / "unzipped") if root.exists()
                  for p in root.rglob(name))


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.symlink(source, target)


def install_packages():
    say("pip install " + " ".join(PACKAGES))
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    if result.returncode != 0:
        say("pip failed (is Internet on in the notebook settings?)")


def prepare_data():
    """Link <anything>/train_source1.tsv etc. into DATA/train and DATA/test."""
    if not find("test_source1.tsv"):
        for archive in find("*.zip"):
            say(f"unzipping {archive}")
            zipfile.ZipFile(archive).extractall(WORKING / "unzipped")
    for split, names in {"train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                                   "train_ground_truth.tsv"],
                         "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]}.items():
        for name in names:
            found = find(name)
            if found:
                link(found[0], DATA / split / name)
            else:
                say(f"not found: {name}")


def reuse_earlier_results():
    """Link files of earlier notebooks (…/work/<step>/<file>) into WORK."""
    for root in (INPUT,):
        for path in root.rglob("work/*/*"):
            if path.is_file():
                link(path, WORK / path.parent.name / path.name)
                say(f"reusing {path}")


def code_dir():
    found = find("run_pipeline.py")
    if not found:
        raise SystemExit("run_pipeline.py not found: add the code dataset as an input")
    return found[0].parent


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


# Extra answers at other thresholds (same model), to test on the leaderboard whether the
# unseen country prefers another threshold; + every test pair's probability.
ASSIGN_PATCH_OLD = """    log(f"wrote {output_dir / 'matching_results.tsv'} and candidate_pairs.tsv")\n"""
ASSIGN_PATCH_NEW = ASSIGN_PATCH_OLD + f"""    try:
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
    """Copy of the code dataset with the assign.py patch (the dataset itself is unchanged)."""
    code = Path("/tmp/aml26_code")
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(code_dir(), code)
    text = (code / "assign.py").read_text(encoding="utf-8")
    if text.count(ASSIGN_PATCH_OLD) != 1:
        raise SystemExit("assign.py patch failed (code dataset changed?)")
    (code / "assign.py").write_text(text.replace(ASSIGN_PATCH_OLD, ASSIGN_PATCH_NEW), encoding="utf-8")
    text = (code / "config.py").read_text(encoding="utf-8")
    for old, new in CONFIG_PATCH.items():
        if text.count(old) != 1:
            raise SystemExit(f"config.py patch failed: {old!r}")
        text = text.replace(old, new)
        say(f"setting {new}")
    (code / "config.py").write_text(text, encoding="utf-8")
    return code


def check_parts():
    """How many candidates per record do the reused candidate parts hold? (must be 30 for the gain)"""
    import pandas as pd
    for part in sorted(WORK.glob("*/candidates_part_*.parquet")):
        rank = pd.read_parquet(part, columns=["candidate_rank"])["candidate_rank"]
        say(f"{part.parent.name}/{part.name}: {len(rank):,} pairs, max candidate_rank {rank.max()}"
            + ("" if rank.max() >= 29 else "   <-- ONLY 15: this part gives no gain"))
    for search in sorted(WORK.glob("*/search_*.parquet")):
        say(f"saved search to combine again: {search.parent.name}/{search.name}")


def run(code, *args):
    """Run the pipeline. A failure does NOT fail the notebook: the notebook still
    finishes, so the finished steps are saved as output and a rerun can reuse them.
    The failure is written to work/FAILED.txt and printed with ### PIPELINE FAILED."""
    command = [sys.executable, "-u", str(code / "run_pipeline.py"), *args,
               "--data-dir", str(DATA), "--work-dir", str(WORK), "--n-jobs", str(N_JOBS)]
    say(" ".join(command))
    time_left = PIPELINE_TIME_LIMIT - (time.time() - NOTEBOOK_START)
    say(f"time limit for the pipeline: {time_left / 3600:.2f} hours")
    try:
        returncode = subprocess.run(command, cwd=code, timeout=time_left).returncode
        problem = f"exit code {returncode}" + (" (-9 = killed, usually out of memory)" if returncode == -9 else "")
    except subprocess.TimeoutExpired:
        returncode, problem = -1, "stopped at the time limit (the finished steps are saved; resume)"
    peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6    # ru_maxrss is in KB
    say(f"peak memory of the pipeline: {peak_gb:.1f} GB; time {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
    if returncode != 0:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "FAILED.txt").write_text(f"{' '.join(args)}: {problem}\n")
        say(f"PIPELINE FAILED ({problem})")
        return False
    say("PIPELINE OK")
    return True


def validate():
    output = WORK / "output"
    validator = find("validate_submission.py")
    if validator and (output / "matching_results.tsv").exists():
        subprocess.run([sys.executable, str(validator[0]),
                        "-m", str(output / "matching_results.tsv"),
                        "-c", str(output / "candidate_pairs.tsv"),
                        "-t", str(DATA / "test")])


def clean_up():
    """Remove links and copies of the input data, so the output only holds our results."""
    for path in WORK.rglob("*"):
        if path.is_symlink():
            path.unlink()
    shutil.rmtree(DATA, ignore_errors=True)
    shutil.rmtree(WORKING / "unzipped", ignore_errors=True)


def main():
    say(f"MODE = {MODE}")
    install_packages()
    prepare_data()
    combine_ce_scores()
    reuse_earlier_results()
    for path in (WORK / "output").glob("*"):    # answers of input notebooks: ours are written fresh
        if path.is_symlink():
            path.unlink()
    check_parts()
    code = patched_code()
    try:
        if MODE == "wide":
            if run(code, "train", "--note", NOTE) and run(code, "test-prep") and run(code, "assign"):
                say("ALL DONE")
        elif MODE == "ensemble":
            if run(code, "train", "--note", NOTE) and run(code, "assign"):
                say("ALL DONE")
        elif MODE == "dry-run":
            run(code, "test-prep", "--dry-run")
        elif MODE.startswith("block "):          # "block <split> <country,country>"
            _, split, countries = MODE.split(" ", 2)
            run(code, "block", "--split", split, "--countries", countries)
        elif MODE in ("train", "test-prep", "assign", "diag", "llm"):
            run(code, MODE, "--note", "kaggle full data") if MODE == "train" else run(code, MODE)
        else:
            raise SystemExit(f"unknown MODE {MODE!r}")
        validate()
    finally:
        clean_up()
    subprocess.run(["du", "-sh", str(WORK)])


main()

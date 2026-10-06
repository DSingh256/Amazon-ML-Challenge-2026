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
MODE = "__MODE__"
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
    else:
        say("PIPELINE OK")


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
    reuse_earlier_results()
    code = code_dir()
    try:
        if MODE == "dry-run":
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

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
MODE = "llm"
# v26 llm7: XLM-R base cross-encoder (as llm3) that ALSO learns from sure TEST pairs
# (self-training / domain adaptation: the test set has a country the training data does not).
# Sure pairs come from aml26-ens1's test probabilities: exclusive p >= 0.95 = match;
# p <= 0.05 among the top 5 of such records, or the top 3 of records whose best p <= 0.02 = no match.
PSEUDO_SRC = "aml26-ens1"
MAX_TRAIN_SECONDS = 5.5 * 3600
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
            if path.is_file() and PSEUDO_SRC not in str(path):
                link(path, WORK / path.parent.name / path.name)
                say(f"reusing {path}")


def original_code_dir():
    found = find("run_pipeline.py")
    if not found:
        raise SystemExit("run_pipeline.py not found: add the code dataset as an input")
    return found[0].parent


CE_OLD = """    model.fine_tune(left_text[learn["s1_row"]], right_text[learn["s23_row"]], learn["label"].to_numpy())
"""
CE_NEW = """    left_learn, right_learn = left_text[learn["s1_row"]], right_text[learn["s23_row"]]
    labels_learn = learn["label"].to_numpy()
    try:
        extra_left, extra_right, extra_labels = pseudo_test_pairs(test_dir, columns)
        left_learn = np.concatenate([left_learn, extra_left])
        right_learn = np.concatenate([right_learn, extra_right])
        labels_learn = np.concatenate([labels_learn, extra_labels]).astype(labels_learn.dtype)
    except Exception as error:
        log(f"PSEUDO failed, learning from the train pairs only: {error!r}")
    model.fine_tune(left_learn, right_learn, labels_learn)
"""
STEP_OLD = "                step += 1\n"
STEP_NEW = STEP_OLD + (f"                if time.time() - start_time > {MAX_TRAIN_SECONDS}:\n"
                       "                    log(f'  time limit for learning: stopped at step {step:,}/{steps:,}')\n"
                       "                    break\n")
PSEUDO_FUNCTION = """

def pseudo_test_pairs(test_dir, columns, n_pos=100_000, n_neg=110_000):
    \"\"\"Sure test pairs (pseudo-labels) from another notebook's test probabilities (env AML26_PSEUDO).\"\"\"
    import os
    table = pd.read_parquet(os.environ["AML26_PSEUDO"])
    s1_row, s23_row = table["s1_row"].to_numpy(), table["s23_row"].to_numpy()
    p = table["probability"].to_numpy()
    del table
    order = np.lexsort((-p, s23_row))
    winner = np.zeros(len(p), dtype=bool)
    winner[order[np.r_[True, s23_row[order][1:] != s23_row[order][:-1]]]] = True
    positive = winner & (p >= 0.95)
    order = np.lexsort((-p, s1_row))
    start = np.r_[True, s1_row[order][1:] != s1_row[order][:-1]]
    group = np.cumsum(start) - 1
    rank = np.empty(len(p), dtype=np.int64)
    rank[order] = np.arange(len(p)) - np.flatnonzero(start)[group]
    n_rows = int(s1_row.max()) + 1
    has_positive = np.zeros(n_rows, dtype=bool)
    has_positive[s1_row[positive]] = True
    best = np.zeros(n_rows)
    best[s1_row[order[start]]] = p[order[start]]
    hard = (p <= 0.05) & (rank < 5) & has_positive[s1_row]
    empty = (rank < 3) & (best[s1_row] <= 0.02)
    rng = np.random.default_rng(7)

    def sample(mask, n):
        idx = np.flatnonzero(mask)
        return rng.choice(idx, n, replace=False) if len(idx) > n else idx

    pos = sample(positive, n_pos)
    neg = np.concatenate([sample(hard, int(n_neg * 0.7)), sample(empty & ~hard, n_neg - int(n_neg * 0.7))])
    idx = np.concatenate([pos, neg])
    labels = np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int8)
    s1 = pd.read_parquet(test_dir / "s1_clean.parquet", columns=columns)
    s23 = pd.read_parquet(test_dir / "s23_clean.parquet", columns=columns)
    left = record_texts(s1.iloc[s1_row[idx]].reset_index(drop=True))
    right = record_texts(s23.iloc[s23_row[idx]].reset_index(drop=True))
    log(f"PSEUDO: + {len(pos):,} sure test matches (of {positive.sum():,}) and {len(neg):,} sure test non-matches "
        f"(hard {hard.sum():,}, empty-record {empty.sum():,} available)")
    return left, right, labels
"""


def code_dir():
    """Copy of the code dataset with the llm7 changes (the dataset itself is unchanged)."""
    code = Path("/tmp/aml26_code")
    if code.exists():
        return code
    shutil.copytree(original_code_dir(), code)
    text = (code / "cross_encoder.py").read_text(encoding="utf-8")
    for old, new in ((CE_OLD, CE_NEW), (STEP_OLD, STEP_NEW)):
        if text.count(old) != 1:
            raise SystemExit(f"cross_encoder.py patch failed: {old[:60]!r}")
        text = text.replace(old, new)
    (code / "cross_encoder.py").write_text(text + PSEUDO_FUNCTION, encoding="utf-8")
    found = [p for p in INPUT.rglob("test_pair_probabilities.parquet") if PSEUDO_SRC in str(p)]
    if not found:
        raise SystemExit(f"test_pair_probabilities.parquet of {PSEUDO_SRC} not found (add it as an input)")
    os.environ["AML26_PSEUDO"] = str(found[0])
    say(f"pseudo-labels from {found[0]}")
    return code


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
    validator = [p for p in find("validate_submission.py") if "aml26_code" not in str(p)]
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

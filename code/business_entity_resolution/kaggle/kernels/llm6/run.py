"""
aml26-llm6: bge-reranker-v2-m3 LLM layer, scores only (GPU), for the ensemble notebook ens2.

    1. llm    : fine-tune BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M parameters, a multilingual
                reranker already trained to score text pairs) as the cross-encoder
                and score the unsure train / test pairs      (~6-7 h on GPU T4 x2)
    2. train  : stage 2 again with the new ce / ce_others_max features   (~1 h)
    3. assign : score the test pairs -> matching_results.tsv + candidate_pairs.tsv  (~0.6 h)
                + the official validator

Everything else is the same code as v18-v20 (dataset amazon-ml-2026-code). The only
changes are the settings below: they are patched into a copy of the code, so the code
dataset is not changed.

Notebook settings: Accelerator = GPU T4 x2, Internet = on, Persistence off.
Inputs (Add Input, all shared by abhishekg14):
    datasets  abhishekg14/amazon-ml-2026-data, abhishekg14/amazon-ml-2026-code
    notebooks abhishekg14/aml26-train3, aml26-test-prep3, aml26-block-train-in, aml26-block-test-in
Run it with "Save Version" -> "Save & Run All (Commit)" (runs in the background, up to 12 h).
Result: Output tab -> matching_results.tsv (the file to upload on Unstop).
The last lines of the log must say "### ALL DONE" and "PASS" (validator).
"""
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

NOTEBOOK_START = time.time()
N_JOBS = 4
PACKAGES = ["anyascii==0.3.3", "rapidfuzz==3.14.6", "sparse_dot_topn==1.2.0"]
NOTE = "llm6: bge-reranker-v2-m3 scores only (50k records x 6, lr 1e-5, 1.5M most unsure test pairs)"

# The v21 settings of the LLM layer (config.py text -> new text). Why:
#   bge-reranker-v2-m3 : XLM-R large size, but already trained as a multilingual pair scorer
#   100,000 records    : large learns ~3.5x slower; 100k x 6 candidates (~700k pairs) fit in ~5 h
#   lr 1e-5            : large models are unstable with the base model's 3e-5
#   p1 0.01 .. 0.99, 3M: the v18/v19 window (~2.4M pairs), scoring is ~3.5x slower too
CONFIG_PATCH = {
    'model: str = "FacebookAI/xlm-roberta-base"': 'model: str = "BAAI/bge-reranker-v2-m3"',
    "train_records: int = 300_000": "train_records: int = 50_000",
    "learning_rate: float = 3e-5": "learning_rate: float = 1e-5",
    "infer_batch: int = 512": "infer_batch: int = 256",
    "max_certainty: float = 0.499": "max_certainty: float = 0.49",
    "score_budget: int = 6_000_000": "score_budget: int = 1_500_000",
}
# Safety: learning stops after this long (the saved model is then used as it is), so a
# slower GPU than expected cannot push the notebook past Kaggle's 12-hour limit.
MAX_TRAIN_SECONDS = 1.75 * 3600
CODE_PATCH = {
    "                step += 1\n":
        "                step += 1\n"
        f"                if time.time() - start_time > {MAX_TRAIN_SECONDS}:\n"
        "                    log(f'  time limit for learning: stopped at step {step:,}/{steps:,} (loss {running:.4f})')\n"
        "                    break\n",
}

# Kaggle stops a notebook after 12 hours and then saves NOTHING, so the pipeline stops earlier.
PIPELINE_TIME_LIMIT = 4.0 * 3600   # GPU quota: must not eat llm5's hours

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"          # links to the challenge .tsv files
WORK = WORKING / "work"          # pipeline results (saved as notebook output)
CODE = Path("/tmp/aml26_code")   # patched copy of the code (not saved)
LOG = WORKING / "llm6_run.log"


def say(message):
    line = f"### {message}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


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
        raise SystemExit("pip failed: turn Internet ON in the notebook settings (needs a phone-verified account)")


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
            if not found:
                raise SystemExit(f"not found: {name} -> add the input abhishekg14/amazon-ml-2026-data")
            link(found[0], DATA / split / name)


def reuse_earlier_results():
    """Link files of the input notebooks (.../work/<step>/<file>) into WORK."""
    count = 0
    for path in INPUT.rglob("work/*/*"):
        if path.is_file():
            link(path, WORK / path.parent.name / path.name)
            count += 1
    say(f"reusing {count} files of the input notebooks")
    for needed in ["train/features.parquet", "test/features.parquet", "models/meta.json",
                   "train/candidates.parquet", "train/s1_clean.parquet", "test/s1_clean.parquet"]:
        if not (WORK / needed).exists():
            raise SystemExit(f"work/{needed} not found -> add the 4 input notebooks "
                             "aml26-train3, aml26-test-prep3, aml26-block-train-in, aml26-block-test-in")


def patch(path, replacements):
    text = path.read_text(encoding="utf-8")
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise SystemExit(f"code patch failed in {path.name}: {old!r} (is the code dataset the latest version?)")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def prepare_code():
    found = find("run_pipeline.py")
    if not found:
        raise SystemExit("run_pipeline.py not found -> add the input abhishekg14/amazon-ml-2026-code")
    shutil.rmtree(CODE, ignore_errors=True)
    shutil.copytree(found[0].parent, CODE)
    patch(CODE / "config.py", CONFIG_PATCH)
    patch(CODE / "cross_encoder.py", CODE_PATCH)
    for new in CONFIG_PATCH.values():
        say(f"setting {new}")


def run(*args):
    """Run one pipeline step, show its output live, stop it at the time limit.
    Returns True when it finished without error."""
    command = [sys.executable, "-u", str(CODE / "run_pipeline.py"), *args,
               "--data-dir", str(DATA), "--work-dir", str(WORK), "--n-jobs", str(N_JOBS)]
    time_left = PIPELINE_TIME_LIMIT - (time.time() - NOTEBOOK_START)
    say(" ".join(command))
    say(f"time left for the pipeline: {time_left / 3600:.2f} hours")
    if time_left < 600:
        say(f"PIPELINE FAILED ({args[0]}: no time left)")
        return False
    process = subprocess.Popen(command, cwd=CODE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, encoding="utf-8", errors="replace")
    timer = threading.Timer(time_left, process.kill)
    timer.start()
    with open(LOG, "a", encoding="utf-8") as log_file:
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
    returncode = process.wait()
    timed_out = not timer.is_alive()
    timer.cancel()
    peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6    # ru_maxrss is in KB
    say(f"peak memory: {peak_gb:.1f} GB; time {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
    if returncode != 0:
        problem = "stopped at the time limit" if timed_out else f"exit code {returncode}" + (
            " (-9 = killed, usually out of memory)" if returncode == -9 else "")
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "FAILED.txt").write_text(f"{' '.join(args)}: {problem}\n")
        say(f"PIPELINE FAILED ({args[0]}: {problem})")
        return False
    say(f"PIPELINE OK ({args[0]})")
    return True


def validate():
    output = WORK / "output"
    validator = find("validate_submission.py")
    if validator and (output / "matching_results.tsv").exists():
        result = subprocess.run([sys.executable, str(validator[0]),
                                 "-m", str(output / "matching_results.tsv"),
                                 "-c", str(output / "candidate_pairs.tsv"),
                                 "-t", str(DATA / "test")], capture_output=True, text=True)
        say("validator:\n" + result.stdout + result.stderr)
        # a copy next to the log, so it is the first thing in the Output tab
        shutil.copy(output / "matching_results.tsv", WORKING / "matching_results.tsv")


def clean_up():
    """Remove links and copies of the input data, so the output only holds our results."""
    for path in WORK.rglob("*"):
        if path.is_symlink():
            path.unlink()
    shutil.rmtree(DATA, ignore_errors=True)
    shutil.rmtree(WORKING / "unzipped", ignore_errors=True)


def main():
    say("aml26-bge: bge-reranker-v2-m3 cross-encoder -> stage 2 -> matching_results.tsv")
    try:
        import torch
        say(f"GPUs: {torch.cuda.device_count()} x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
        if torch.cuda.device_count() == 0:
            raise SystemExit("no GPU: set Accelerator = GPU T4 x2 in the notebook settings")
    except ImportError:
        pass
    install_packages()
    prepare_data()
    reuse_earlier_results()
    prepare_code()
    try:
        # llm6 = scores only; the ensemble notebook (ens2) does train + assign on CPU
        if run("llm"):
            say("ALL DONE: work/train + work/test ce_scores.parquet saved")
    finally:
        clean_up()
    subprocess.run(["du", "-sh", str(WORK)])


main()

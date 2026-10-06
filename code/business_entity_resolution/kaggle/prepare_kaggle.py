"""
Build the folders that are uploaded to Kaggle (see RUNBOOK.md).

    <out>/data/                 private dataset <user>/amazon-ml-2026-data (challenge files, zipped)
    <out>/code/                 private dataset <user>/amazon-ml-2026-code (src/*.py + *.json, requirements.txt)
    <out>/kernels/<mode>/       one private notebook per step (kernel_run.py with MODE set)

Usage:
    python kaggle/prepare_kaggle.py --user abhishekg14 --resource-dir D:/AmazonML/student_resource \
                                    --out D:/AmazonML/kaggle_build
    python kaggle/prepare_kaggle.py ... --code-only      # after a code change
"""
import argparse
import json
import shutil
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
DATA_SLUG = "amazon-ml-2026-data"
CODE_SLUG = "amazon-ml-2026-code"
KERNEL_PREFIX = "aml26-"
# notebook name -> (mode, notebooks whose output it needs as input).
# dry-run saves the cleaned test tables; test-prep and assign reuse them.
# The "-resume" notebooks continue a run that stopped (crash, time limit): they get the
# saved output of the stopped notebook as input and skip everything that is finished.
KERNELS = {
    "dry-run": ("dry-run", []),
    "train": ("train", []),
    "test-prep": ("test-prep", ["dry-run"]),
    "assign": ("assign", ["dry-run", "train2", "test-prep"]),   # train2 = the v6 train run
    "train-resume": ("train", ["train2"]),
    "diag": ("diag", []),         # blocking diagnostic on part of the train data (India)
    "train2": ("train", []),      # a fresh train run next to a still-running "train" (after a code fix)
    "test-prep-resume": ("test-prep", ["dry-run", "test-prep"]),
    # v9: the wider blocking is ~3.4x the search work, too much for one 12-hour notebook, so
    # every split's blocking is spread over notebooks that run at the same time (by country
    # label; a country missing from these lists is simply blocked by train3 / test-prep3).
    "block-train-in": ("block train India", []),
    "block-train-us": ("block train US", []),
    "block-test-in": ("block test India", []),
    "block-test-fr-us": ("block test France,US", []),
    # block-train-us ran out of memory after its searches; the resume notebook finished it
    "block-train-us-resume": ("block train US", ["block-train-us"]),
    "train3": ("train", ["block-train-in", "block-train-us", "block-train-us-resume"]),
    "test-prep3": ("test-prep", ["block-test-in", "block-test-fr-us"]),
    # the cleaned test tables are only in the block notebooks' output (inputs are not re-saved)
    "assign3": ("assign", ["train3", "test-prep3", "block-test-in"]),
    # the "LLM layer" (cross_encoder.py) on a GPU, then stage 2 again with its scores
    "llm": ("llm", ["train3", "test-prep3", "block-train-in", "block-test-in"]),
    "train4": ("train", ["train3", "block-train-in", "llm"]),
    "assign4": ("assign", ["train4", "test-prep3", "block-test-in", "llm"]),
    # v16: the same train3 features, model without the legal-form bit numbers (no LLM scores)
    "train5": ("train", ["train3", "block-train-in"]),
    "assign5": ("assign", ["train5", "test-prep3", "block-test-in"]),
    # v17: the train3 model again, only the decoder rule changed (threshold decoder)
    "assign3t": ("assign", ["train3", "test-prep3", "block-test-in"]),
    # v18: the LLM layer learns from 150,000 records (hardest 6 candidates each) instead of 40,000
    "llm2": ("llm", ["train3", "test-prep3", "block-train-in", "block-test-in"]),
    "train6": ("train", ["train3", "block-train-in", "llm2"]),
    "assign6": ("assign", ["train6", "test-prep3", "block-test-in", "llm2"]),
    # v19: the LLM layer learns from 300,000 records (hardest 6 candidates each)
    "llm3": ("llm", ["train3", "test-prep3", "block-train-in", "block-test-in"]),
    "train7": ("train", ["train3", "block-train-in", "llm3"]),
    "assign7": ("assign", ["train7", "test-prep3", "block-test-in", "llm3"]),
    # v20: as v19, and the LLM layer also scores pairs with stage-1 probability 0.001 .. 0.999
    "llm4": ("llm", ["train3", "test-prep3", "block-train-in", "block-test-in"]),
    "train8": ("train", ["train3", "block-train-in", "llm4"]),
    "assign8": ("assign", ["train8", "test-prep3", "block-test-in", "llm4"]),
}
GPU_KERNELS = {"llm", "llm2", "llm3", "llm4"}


def dataset_metadata(folder, user, slug, title):
    (folder / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{user}/{slug}", "licenses": [{"name": "other"}]}, indent=2))


def build_data(resource_dir, out, user):
    """Zip the challenge files (about 4x smaller to upload) + the validator."""
    folder = out / "data"
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "student_resource.zip"
    if archive.exists():
        print(f"{archive} already exists, not rebuilt")
    else:
        files = sorted((resource_dir / "dataset").rglob("*.tsv")) + [resource_dir / "utils" / "validate_submission.py"]
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in files:
                print(f"  adding {path.relative_to(resource_dir)}")
                z.write(path, path.relative_to(resource_dir))
    dataset_metadata(folder, user, DATA_SLUG, "Amazon ML 2026 data")
    print(f"data dataset: {folder} ({archive.stat().st_size / 1e6:,.0f} MB)")


def build_code(out, user):
    folder = out / "code"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)      # Windows may keep the emptied folder
    for path in [*(PROJECT / "src").glob("*.py"), *(PROJECT / "src").glob("*.json")]:
        shutil.copy(path, folder / path.name)
    shutil.copy(PROJECT / "requirements.txt", folder / "requirements.txt")
    dataset_metadata(folder, user, CODE_SLUG, "Amazon ML 2026 code")
    print(f"code dataset: {folder}")


def build_kernels(out, user):
    template = (HERE / "kernel_run.py").read_text()
    for name, (mode, needs) in KERNELS.items():
        folder = out / "kernels" / name
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "run.py").write_text(template.replace("__MODE__", mode))
        slug = KERNEL_PREFIX + name
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{slug}",
            "title": slug,
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": name in GPU_KERNELS,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [f"{user}/{DATA_SLUG}", f"{user}/{CODE_SLUG}"],
            "competition_sources": [],
            "kernel_sources": [f"{user}/{KERNEL_PREFIX}{name}" for name in needs],
        }, indent=2))
        if name in GPU_KERNELS:
            metadata = json.loads((folder / "kernel-metadata.json").read_text())
            metadata["machine_shape"] = "NvidiaTeslaT4"       # 2 x T4
            (folder / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
        print(f"notebook {user}/{slug}: {folder}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, help="Kaggle username")
    parser.add_argument("--resource-dir", default="D:/AmazonML/student_resource")
    parser.add_argument("--out", default="D:/AmazonML/kaggle_build")
    parser.add_argument("--code-only", action="store_true",
                        help="rebuild only the code dataset and the notebooks (not the 1 GB data zip)")
    args = parser.parse_args()
    out = Path(args.out)
    if not args.code_only:
        build_data(Path(args.resource_dir), out, args.user)
    build_kernels(out, args.user)
    build_code(out, args.user)


if __name__ == "__main__":
    main()

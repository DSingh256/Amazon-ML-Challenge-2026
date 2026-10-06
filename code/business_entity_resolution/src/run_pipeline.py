"""
The whole pipeline with one command. Steps whose output file already exists
are skipped (delete the file, or use --redo, to run a step again).

    train     : clean -> blocking -> features -> train the model     (<work>/train, <work>/models)
    test-prep : clean -> blocking -> features for the test split     (<work>/test)
    assign    : score test pairs and write the final files          (<work>/output)
    all       : train, test-prep, assign
    llm       : the "LLM layer" (cross_encoder.py, GPU): scores of the unsure pairs of train and test;
                train + assign then use them when <work>/<split>/ce_scores.parquet exists
    block     : clean + blocking of some countries of one split only (--split, --countries);
                several notebooks share the blocking of a big split, then "train" /
                "test-prep" with their outputs as input reuse the saved parts

On Kaggle, "train" and "test-prep" can run at the same time in two notebooks;
a third small notebook then runs "assign" with both outputs.

Examples:
    python src/run_pipeline.py all --data-dir D:/AmazonML/work/sample_data --work-dir D:/AmazonML/work/sample_run
    python src/run_pipeline.py test-prep --data-dir /kaggle/working/data --work-dir /kaggle/working/work --n-jobs 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

from assign import write_outputs
from features import make_features
from io_utils import log
from run_blocking import run_blocking
from train import run_training

STEP_OUTPUTS = {"blocking": "candidates.parquet", "features": "features.parquet"}


def prepare_split(data_dir, split, work_dir, n_jobs, redo, dry_run):
    """clean + blocking + features for one split (each step skipped if done)."""
    split_dir = Path(work_dir) / split
    if dry_run:
        run_blocking(data_dir, split, split_dir, n_jobs, dry_run=True)
        return False
    if redo or not (split_dir / STEP_OUTPUTS["blocking"]).exists():
        if redo:
            for old in [*split_dir.glob("candidates_part_*.parquet"), *split_dir.glob("search_*.parquet")]:
                old.unlink()          # saved results of an earlier run
        log(f"=== {split}: clean + blocking ===")
        run_blocking(data_dir, split, split_dir, n_jobs)
    else:
        log(f"=== {split}: blocking already done ===")
    if redo or not (split_dir / STEP_OUTPUTS["features"]).exists():
        log(f"=== {split}: features ===")
        make_features(split_dir, split, data_dir)
    else:
        log(f"=== {split}: features already done ===")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "test-prep", "assign", "all", "diag", "block", "prep-train", "llm"])
    parser.add_argument("--data-dir", required=True, help="folder that contains train/ and test/")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--n-jobs", type=int, default=4, help="CPU cores for cleaning")
    parser.add_argument("--redo", action="store_true", help="run every step again (cleaning stays cached)")
    parser.add_argument("--dry-run", action="store_true", help="only clean and print the blocking work")
    parser.add_argument("--transfer-check", action="store_true", help="train: also test on unseen countries")
    parser.add_argument("--note", default="", help="train: what changed, for results.md")
    parser.add_argument("--split", choices=["train", "test"], help="block: which split")
    parser.add_argument("--countries", help="block: comma-separated country labels")
    parser.add_argument("--method", choices=["auto", "threshold", "expected"], default="auto")
    args = parser.parse_args()
    work_dir = Path(args.work_dir)

    if args.mode == "diag":            # why are true pairs missing from the candidates? (diag_blocking.py)
        from diag_blocking import main as diagnose
        diagnose(["--data-dir", args.data_dir, "--work-dir", str(work_dir), "--n-jobs", str(args.n_jobs), "--variants-only"])
        return

    if args.mode == "llm":             # the "LLM layer" on a GPU (cross_encoder.py); needs train + test done
        from cross_encoder import run_cross_encoder
        log("=== cross-encoder: learn, then score the unsure pairs ===")
        run_cross_encoder(work_dir, args.data_dir)
        log("pipeline finished")
        return

    if args.mode == "block":
        countries = [c.strip() for c in args.countries.split(",")]
        log(f"=== {args.split}: clean + blocking of {', '.join(countries)} ===")
        run_blocking(args.data_dir, args.split, work_dir / args.split, args.n_jobs, countries=countries)
        log("pipeline finished")
        return

    if args.mode == "prep-train":      # clean + blocking + features of the train split only (see below)
        prepare_split(args.data_dir, "train", work_dir, args.n_jobs, args.redo, False)
        return

    if args.mode in ("train", "all"):
        if args.dry_run:
            prepare_split(args.data_dir, "train", work_dir, args.n_jobs, args.redo, True)
        else:
            # The features step leaves ~12 GB that Python never gives back to the system,
            # so it runs in its own process and the model starts with free memory.
            subprocess.run([sys.executable, "-u", __file__, "prep-train", "--data-dir", args.data_dir,
                            "--work-dir", str(work_dir), "--n-jobs", str(args.n_jobs)]
                           + (["--redo"] if args.redo else []), check=True)
            log("=== train: model ===")
            run_training(work_dir, args.data_dir, args.transfer_check, args.note)

    if args.mode in ("test-prep", "all"):
        prepare_split(args.data_dir, "test", work_dir, args.n_jobs, args.redo, args.dry_run)

    if args.mode in ("assign", "all") and not args.dry_run:
        log("=== assign: final files ===")
        write_outputs(work_dir / "test", work_dir / "models", work_dir / "output", args.method)

    log("pipeline finished")


if __name__ == "__main__":
    main()

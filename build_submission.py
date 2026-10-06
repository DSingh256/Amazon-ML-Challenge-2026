"""
Build <team_name>_submission.zip in the required challenge structure:

    output/matching_results.tsv, output/candidate_pairs.tsv
    code/business_entity_resolution/  (src/, README.md, requirements.txt, kaggle/, tests/)
    Documentation_template.md

Re-stages everything from the repository (fresh copy, no work files, no caches), runs the
syntax + unit tests on the staged copy, and zips it. Run again after ANY change to outputs
or code:

    python build_submission.py                      # 404_not_found_submission.zip
    python build_submission.py --team "Other Name"  # other_name_submission.zip

The team name and the final scores (holdout 0.9874, LB 0.983) live directly in
Documentation_template.md; edit them there. The v16 experiment switches (BLOCKING.rescue)
are forced OFF in the staged copy, because the submitted outputs were produced with the
final wide30 + rival configuration (30 candidates per record, rescue off).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEAM_DEFAULT = "404 Not found"
REVERT = [
    ("src/config.py", "rescue: int = 2", "rescue: int = 0"),
    ("src/stage2.py",
     'STAGE2_COLUMNS = ["p1", "p1_rank", "p1_others_max", "p1_others_sum", "n_sure_others",\n'
     '                  "p1_gap_best", "p1_log_odds", "p1_log_odds_best",\n'
     '                  "sib_addr_support", "sib_name_support"] + RIVAL_COLUMNS',
     'STAGE2_COLUMNS = ["p1", "p1_rank", "p1_others_max", "p1_others_sum", "n_sure_others",\n'
     '                  "sib_addr_support", "sib_name_support"] + RIVAL_COLUMNS'),
]


def run(cmd, cwd):
    result = subprocess.run([sys.executable, *cmd], cwd=cwd, capture_output=True, text=True,
                            env={**os.environ, "PYTHONUTF8": "1"})
    if result.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=TEAM_DEFAULT, help="only used for the zip file name")
    parser.add_argument("--out", default=None, help="zip path (default <root>/<slug>_submission.zip)")
    args = parser.parse_args()
    slug = args.team.lower().replace(" ", "_")
    zip_path = Path(args.out) if args.out else ROOT / f"{slug}_submission.zip"

    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "stage"
        (stage / "output").mkdir(parents=True)
        (stage / "code").mkdir()
        shutil.copy2(ROOT / "output" / "matching_results.tsv", stage / "output" / "matching_results.tsv")
        shutil.copy2(ROOT / "output" / "candidate_pairs.tsv", stage / "output" / "candidate_pairs.tsv")
        shutil.copy2(ROOT / "Documentation_template.md", stage / "Documentation_template.md")
        shutil.copytree(ROOT / "code" / "business_entity_resolution", stage / "code" / "business_entity_resolution",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "work", "out_*"))
        code = stage / "code" / "business_entity_resolution"
        for rel, old, new in REVERT:
            path = code / rel
            text = path.read_text(encoding="utf-8")
            if old in text:
                path.write_text(text.replace(old, new), encoding="utf-8")
            elif new not in text:
                raise SystemExit(f"neither {old!r} nor {new!r} found in {rel}")

        # sanity: syntax + unit tests inside the staged copy
        run(["-m", "py_compile", *[str(p) for p in sorted(code.glob("src/*.py"))],
             *[str(p) for p in sorted(code.glob("tests/*.py"))]], code)
        run(["tests/test_normalize.py"], code)
        run(["tests/test_enhancements.py"], code)

        size = sum(f.stat().st_size for f in stage.rglob("*") if f.is_file())
        print(f"staged {size / 1e6:.1f} MB; zipping -> {zip_path.name}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for file in sorted(stage.rglob("*")):
                if file.is_file():
                    z.write(file, file.relative_to(stage).as_posix())
    print(f"done: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

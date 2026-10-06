# Business Entity Resolution — team PI (Amazon ML Challenge 2026)

For every Source 1 record, find all Source 2 / Source 3 records of the same business.
Pipeline: clean text → candidate search (char 3-gram TF-IDF) → pair features →
LightGBM match probability → one-owner rule + per-record decision → output files.

No external data, APIs or geocoding. Only the challenge files are used.

## Setup

```bash
pip install -r requirements.txt       # Python 3.11+
```

Data layout expected by `--data-dir`: `<data-dir>/train/train_source{1,2,3}.tsv`,
`<data-dir>/train/train_ground_truth.tsv`, `<data-dir>/test/test_source{1,2,3}.tsv`
(= `student_resource/dataset`).

## Reproduce both output files (one command)

```bash
python src/run_pipeline.py all --data-dir <student_resource>/dataset --work-dir work --n-jobs 4
```

Results: `work/output/matching_results.tsv` and `work/output/candidate_pairs.tsv`.

Or step by step (each step skips work whose output file already exists):

```bash
python src/run_pipeline.py train     --data-dir DATA --work-dir work   # clean, blocking, features, model
python src/run_pipeline.py test-prep --data-dir DATA --work-dir work   # clean, blocking, features (test)
python src/run_pipeline.py assign    --data-dir DATA --work-dir work   # score + write output/
python src/run_pipeline.py test-prep --data-dir DATA --work-dir work --dry-run   # only print blocking work
```

Validate:

```bash
python <student_resource>/utils/validate_submission.py -m work/output/matching_results.tsv -c work/output/candidate_pairs.tsv -t <student_resource>/dataset/test
```

We ran the full data on Kaggle CPU notebooks (4 cores, 30 GB RAM), see
`kaggle/RUNBOOK.md`. The same `run_pipeline.py` runs there through `kaggle/kernel_run.py`.

## Code (`src/`)

| file | step | what it does |
|---|---|---|
| `io_utils.py` | S0 | read/write .tsv (tab, no quoting), ids ↔ row numbers, fixed hash buckets |
| `text_rules.py`, `normalize.py` | S1 | transliteration to ASCII, lower case, abbreviations, legal forms, numbers/PIN/ZIP |
| `blocking.py`, `run_blocking.py` | S2 | char 3-gram TF-IDF top-k search per country, forward S1→S2, S1→S3 and reverse S2/3→S1 |
| `features.py` | S3 | name / address / legal-form / number similarities, rank and competition features, search_margin |
| `train.py` | S4, S8 | LightGBM, holdout macro F0.5, breakdown report, error list |
| `stage2.py` | S5 | collective features (siblings, rival records of the same name / address) |
| `rival.py` | S5 | same-name / same-address rival features for unsure pairs (the ens3 upgrade) |
| `assign.py` | S6, S7 | each S2/S3 record goes to at most one S1; threshold or expected-F0.5 decision |
| `evaluate.py` | S8 | official macro F0.5, blocking recall report |
| `config.py` | – | all parameters |
| `make_sample.py` | – | small train sample for laptop experiments |

Tests of the text cleaning: `python tests/test_normalize.py`.
Tests of the enhancements (rescue, rival, gap features): `python tests/test_enhancements.py`.
End-to-end smoke test without challenge data:
`py -3.12 tests/make_synthetic.py --out-dir work/smoke_data && python src/run_pipeline.py all --data-dir work/smoke_data --work-dir work/smoke_run`.

`country` is only used to group records for the search (any label, e.g. France,
works the same way); it is never a model feature.

## Hardware and run time (full data, Kaggle CPU 4 cores / 30 GB)

| stage | train | test |
|---|---|---|
| clean | TODO | TODO |
| blocking | TODO | TODO |
| features | TODO | TODO |
| model training | TODO | – |
| assign | – | TODO |

Random seed: 42 (`config.py`). Train/holdout split: fixed CRC32 hash of the id.

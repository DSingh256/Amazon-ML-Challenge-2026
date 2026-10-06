# Kaggle runbook (S9)

Everything is private. `K` = the Kaggle CLI:
`C:\Users\Admin\AppData\Roaming\Python\Python314\Scripts\kaggle.exe`

## 1. Build the upload folders (laptop)

```bash
python kaggle/prepare_kaggle.py --user abhishekg14     # data zip (1.1 GB), code, 4 notebooks
python kaggle/prepare_kaggle.py --user abhishekg14 --code-only   # after a code change
```

Output goes to `D:/AmazonML/kaggle_build/`.

## 2. Upload the datasets (once for data, then after every code change)

Run the CLI from inside `D:/AmazonML/kaggle_build` with RELATIVE paths: on Windows
the CLI crashes on `-p D:/...` (it builds a bad cache path from the drive letter).

```bash
cd D:/AmazonML/kaggle_build
K datasets create -p data                 # private by default
K datasets create -p code
K datasets version -p code -m "what changed"   # code updates
K datasets status abhishekg14/amazon-ml-2026-data                   # wait for "ready"
```

## 3. Run the notebooks

Each notebook is one `run.py` with a different `MODE`. A push starts a committed
run on Kaggle's machines, so the laptop can be switched off.

| order | notebook | does | uses the output of |
|---|---|---|---|
| 1 | `aml26-dry-run` | cleans test, prints the blocking work (no search) | – |
| 2a | `aml26-train2` | clean + blocking + features + LightGBM on full train | – |
| 2b | `aml26-test-prep` | blocking + features on test | dry-run |
| 3 | `aml26-assign` | final `output/matching_results.tsv` + `candidate_pairs.tsv`, validator | dry-run, train2, test-prep |

2a and 2b run at the same time. (`aml26-train` is the first try, code v3: it ran out of
memory; its model does not fit the v6 features, do not use it.) Train and test-prep must
run the SAME code version: the features (e.g. hashed 3-letter pieces) must be made the same way.

`aml26-bench` (kernels/bench) timed the search settings on 20k India queries x 2.3M documents;
the fastest (documents in blocks of 100k, work_budget 1e9: 202 M/s instead of 49 M/s) is
in `config.py`.

```bash
K kernels push -p kernels/dry-run
K kernels status abhishekg14/aml26-dry-run
K kernels output abhishekg14/aml26-dry-run -p D:/AmazonML/work/kaggle/dry-run   # log + files
```

**Time check after the dry run:** the log prints `total work N million` for every
search. The search runs at about 200 million per second (4 threads, blocks of 100k docs);
the full test set is ~1.6 million million = ~2.5 hours of search.
If the total is too slow for the 12-hour limit, lower `min_query_ngrams` in
`config.py` (12 → 10 → 8), push the code again, and repeat.

## 4. Get the submission files

```bash
K kernels output abhishekg14/aml26-assign -p D:/AmazonML/work/kaggle/assign
```

`work/output/matching_results.tsv` and `candidate_pairs.tsv` go into the zip (S10).
Check the validator lines at the end of the notebook log.

## If a notebook stops (crash, memory, 12-hour limit): resume

What is saved as notebook output survives. Every step skips work whose file exists, and
blocking also saves each country's candidates (`candidates_part_<country>.parquet`), so
only the country that was running is lost.

1. Look at the end of the log (`kernels output ... --file-pattern ".*\.log$"`, set
   `PYTHONUTF8=1` on Windows). `### PIPELINE FAILED` and `work/FAILED.txt` mark a crash.
2. Push the matching resume notebook. It gets the stopped notebook as input:

```bash
K kernels push -p kernels/train-resume        # after aml26-train stopped
K kernels push -p kernels/test-prep-resume    # after aml26-test-prep stopped
```

3. `aml26-assign` must then read the resume notebook AS WELL AS the stopped one (a
   notebook's output only holds the files it made itself; the reused files are links that
   are removed at the end): edit `kernels/assign/kernel-metadata.json` (`kernel_sources`:
   add `aml26-train-resume` and/or `aml26-test-prep-resume`; keep the others) before pushing assign.
   Blocking also saves every finished search (`search_<country>_<retriever>_fwd2/fwd3/rev.parquet`,
   deleted when the country is done), so even a crash in the middle of a big country
   loses at most one search.
4. A resume can be lost only if Kaggle deletes the output of a run killed by the time limit
   (not verified). Then the resume starts from the last finished step of the dry-run.

## Notes

- Safety limits in `run.py`: the pipeline is stopped at 11 h 15 min so the notebook can
  still save its output (the log says `stopped at the time limit`: push the resume notebook).
  The log shows the memory in use on every line (`[min  GB]`) and the peak memory at the end;
  exit code -9 = killed for using too much memory (limit ~30 GB).
- Model training log: loss and AUC of train and valid every 50 rounds, where early stopping
  stopped and the best round; all of it is saved in `models/training_curve.csv|png` and
  `models/metrics.json`.

- If `normalize.py` changes, run dry-run again (it holds the cleaned test tables).
- Changing only features/model code: push the code, then rerun train and assign
  (test-prep only when features changed).
- The train notebook also writes `train/blocking_report.txt` (recall on full data),
  `train/holdout_report.txt`, `train/errors.tsv` and `results.md`.

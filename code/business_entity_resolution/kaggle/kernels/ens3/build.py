"""Builds kernels/ens3/run.py from kernels/ens2/run.py + rival.py (a script kernel uploads one file)."""
import base64
from pathlib import Path

here = Path(__file__).parent
src = (here.parent / "ens2" / "run.py").read_text(encoding="utf-8")
rival = base64.b64encode((here / "rival.py").read_bytes()).decode()


def sub(old, new, count=1):
    global src
    if src.count(old) != count:
        raise SystemExit(f"build: expected {count}x {old[:70]!r}, found {src.count(old)}")
    src = src.replace(old, new)


sub('"""\naml26-ens2: the FINAL combination, CPU only. v27.',
    '"""\naml26-ens3 = aml26-ens2 + SAME-NAME / SAME-ADDRESS RIVAL features in stage 2 (rival.py, found with\n'
    'the local ctx2 re-scorer: +0.0015 honest on the holdout) + the holdout pairs saved (work/dec) for\n'
    'local decoder tuning. Everything else as ens2, so ens3 vs ens2 is a clean A/B. v28.\n\n'
    'aml26-ens2: the FINAL combination, CPU only. v27.')
sub('NOTE = "v27 ens2: wide30 (30 candidates) + mean cross-encoder of all llm notebooks + pseudo-labels"',
    'NOTE = "v28 ens3: ens2 + same-name/same-address rival features in stage 2"\n'
    f'RIVAL_B64 = "{rival}"')
# gen uses the patched code (rival columns) and keeps them in the pseudo pairs
sub("    sys.path.insert(0, str(code_dir()))\n    from stage2 import collective_features\n",
    '    sys.path.insert(0, "/tmp/aml26_code")\n    from stage2 import collective_features\n    from rival import RIVAL_COLUMNS\n')
sub('            part = part[meta["features"]].copy()\n',
    '            part = part[list(dict.fromkeys(meta["features"] + RIVAL_COLUMNS))].copy()\n')
# stage2.py patch + rival.py + holdout save
sub('''    for name, old, new in (("train.py", TRAIN_OLD, TRAIN_NEW), ("assign.py", ASSIGN_OLD, ASSIGN_NEW)):''',
    '''    import base64
    (code / "rival.py").write_bytes(base64.b64decode(RIVAL_B64))
    for name, old, new in (("train.py", TRAIN_OLD, TRAIN_NEW), ("assign.py", ASSIGN_OLD, ASSIGN_NEW),
                           ("stage2.py", S2_COLS_OLD, S2_COLS_NEW), ("stage2.py", S2_OLD, S2_NEW),
                           ("train.py", DEC_OLD, DEC_NEW)):''')
sub('''def patched_code():''', '''S2_COLS_OLD = """                  "sib_addr_support", "sib_name_support"]\n"""
S2_COLS_NEW = """                  "sib_addr_support", "sib_name_support"] + __import__("rival").RIVAL_COLUMNS\n"""
S2_OLD = """    frames = []\n    for order, _, _, numbers in prepared:\n"""
S2_NEW = """    from rival import rival_features, release
    for (s1_row, s23_row, _), (order, _, q, numbers) in zip(parts, prepared):
        numbers.update(rival_features(s1_row[order], s23_row[order], q, len(s23_names)))
    release()
""" + S2_OLD
DEC_OLD = """    result = evaluate(model, columns, holdout, holdout_rows, truth, "HOLDOUT")\n"""
DEC_NEW = DEC_OLD + """    try:
        dec_dir = work_dir / "dec"
        dec_dir.mkdir(parents=True, exist_ok=True)
        hp = holdout[["s1_row", "s23_row", "label"]].copy()
        hp["probability"] = model.predict(holdout[columns]).astype(np.float32)
        hp.to_parquet(dec_dir / "holdout_pairs.parquet", index=False)
        truth[truth["s1_row"].isin(holdout_rows)].to_parquet(dec_dir / "holdout_truth.parquet", index=False)
        pd.DataFrame({"s1_row": holdout_rows, "country": s1["country"].to_numpy()[holdout_rows]}).to_parquet(
            dec_dir / "holdout_rows.parquet", index=False)
        log(f"DEC: saved {len(hp):,} holdout pairs to {dec_dir}")
    except Exception as error:
        log(f"DEC save failed: {error!r}")
"""


def patched_code():''')
sub('''        text = (code / name).read_text(encoding="utf-8")
        if text.count(old) != 1:''', '''        text = (code / name).read_text(encoding="utf-8")
        if text.count(old) != 1:''')
sub('    say(f"aml26-ens2: {NOTE}")\n', '    say(f"aml26-ens3: {NOTE}")\n    os.environ["AML26_DATA"] = str(DATA)\n')
(here / "run.py").write_text(src, encoding="utf-8")
print("built", len(src), "chars")

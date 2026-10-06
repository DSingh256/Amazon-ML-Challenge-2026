"""Builds kernels/ens3a/run.py from kernels/ens3/run.py: ASSIGN ONLY with ens3b's trained model
(aml26-ens3 version 2 output) + memory patches (ens3b's assign was killed at 31 GB)."""
from pathlib import Path
here = Path(__file__).parent
src = (here.parent / "ens3" / "run.py").read_text(encoding="utf-8")


def sub(old, new):
    global src
    if src.count(old) != 1:
        raise SystemExit(f"build: expected 1x {old[:70]!r}, found {src.count(old)}")
    src = src.replace(old, new)


sub('"""\naml26-ens3 = ', '"""\naml26-ens3a: ASSIGN ONLY with the model trained by aml26-ens3 v29 (ens3b: holdout 0.9874) whose\n'
    'assign ran out of memory; stage-2 frame built column by column, CE columns assigned in place.\n\naml26-ens3 = ')
sub('NOTE = "v29 ens3b:', 'NOTE = "v30 ens3a (assign of v29 ens3b):')

# memory patches
sub('''S2_NEW = """    from rival import rival_features, release
    for (s1_row, s23_row, _), (order, _, q, numbers) in zip(parts, prepared):
        numbers.update(rival_features(s1_row[order], s23_row[order], q, len(s23_names)))
    release()
""" + S2_OLD''', '''S2_NEW = """    from rival import rival_features, release
    for (s1_row, s23_row, _), (order, _, q, numbers) in zip(parts, prepared):
        numbers.update(rival_features(s1_row[order], s23_row[order], q, len(s23_names)))
    release()
    import gc as _gc, ctypes as _ct
    _gc.collect()
    try:
        _ct.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
""" + S2_OLD
FR_OLD = """    for order, _, _, numbers in prepared:
        frame = pd.DataFrame({k: np.asarray(numbers[k], dtype=np.float32) for k in STAGE2_COLUMNS})
        back = np.empty_like(order)
        back[order] = np.arange(len(order))
        frames.append(frame.iloc[back].reset_index(drop=True))
"""
FR_NEW = """    for order, _, _, numbers in prepared:
        cols = {}
        for k in STAGE2_COLUMNS:            # straight into input order; no second full frame
            col = np.empty(len(order), dtype=np.float32)
            col[order] = numbers.pop(k)
            cols[k] = col
        numbers.clear()
        frames.append(pd.DataFrame(cols))
        del cols
"""
CE_OLD = """        extra = pd.concat([extra, ce_features(pairs["s1_row"], pairs["s23_row"], scores)], axis=1)
"""
CE_NEW = """        _ce = ce_features(pairs["s1_row"], pairs["s23_row"], scores)
        for _c in _ce.columns:
            extra[_c] = _ce[_c].to_numpy()
        del _ce
"""''')
sub('''                           ("train.py", DEC_OLD, DEC_NEW)):''',
    '''                           ("train.py", DEC_OLD, DEC_NEW), ("stage2.py", FR_OLD, FR_NEW),
                           ("assign.py", CE_OLD, CE_NEW)):''')

# main: prepare, the trained model, assign
sub('''        prepare()
        code = patched_code()
        if (gen_in_child() and pipeline(code, "train", "--note", NOTE) and pipeline(code, "assign")):''',
    '''        prepare()
        code = patched_code()
        meta = [p for p in INPUT.rglob("work/models/meta.json") if "aml26-ens3" in str(p)]
        if not meta:
            raise SystemExit("aml26-ens3 models not found")
        for path in meta[0].parent.iterdir():
            link(path, WORK / "models" / path.name)
        say(f"models <- {meta[0].parent}: {sorted(p.name for p in meta[0].parent.iterdir())}")
        if pipeline(code, "assign"):''')
(here / "run.py").write_text(src, encoding="utf-8")
print("built", len(src))

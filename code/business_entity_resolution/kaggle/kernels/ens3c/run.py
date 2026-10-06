"""
aml26-ens3a: ASSIGN ONLY with the model trained by aml26-ens3 v29 (ens3b: holdout 0.9874) whose
assign ran out of memory; stage-2 frame built column by column, CE columns assigned in place.

aml26-ens3 = aml26-ens2 + SAME-NAME / SAME-ADDRESS RIVAL features in stage 2 (rival.py, found with
the local ctx2 re-scorer: +0.0015 honest on the holdout) + the holdout pairs saved (work/dec) for
local decoder tuning. Everything else as ens2, so ens3 vs ens2 is a clean A/B. v28.

aml26-ens2: the FINAL combination, CPU only. v27.
    base      : aml26-wide30 (30 candidates per record instead of 15: more true matches reachable)
    LLM layer : mean cross-encoder score of EVERY aml26-llm<N> notebook in the inputs (llm2..llm7)
    self-training: sure test pairs of wide30's model become extra stage-2 learning pairs (as pseudo1)
(below: the pseudo1 description, which still holds)

aml26-pseudo1: self-training (pseudo-labels) on the TEST set, CPU only. v25.

The test set has a country the training data does not have (France). The model never saw
a French match, so its French probabilities are less right. Self-training lets the model
see test data: the test pairs the current model (SRC = aml26-ens1) is almost sure about
become extra training examples for stage 2.
    positive : the pair keeps its Source 2/3 record after exclusivity and p >= POS
    negative : p <= NEG
No hand labels, no external data, no country rule: every test record is treated the same.

Steps
    gen    : SRC's models score all test pairs (stage 1 -> collective -> cross-encoder -> stage 2,
             exactly like assign.py), pick the sure pairs, write work/pseudo/pseudo_pairs.parquet
             with every stage-2 feature + label
    train  : train.py patched: the pseudo pairs are added to the LEARNING part only (early
             stopping and the holdout stay real train labels, so the holdout still tells
             whether India / US got worse)
    assign : the new model -> matching_results.tsv (+ t0.5/0.6/0.8 files, test probabilities)
"""
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

NOTEBOOK_START = time.time()
SRC = "aml26-wide30"
NOTE = "v31 ens3c (ens3b + llm7 CE, seed 7, memory patches): ens2 + rival features + 12% pseudo pairs (float32; v28 ran out of memory)"
RIVAL_B64 = "IiIiClNhbWUtbmFtZSAvIHNhbWUtYWRkcmVzcyBSSVZBTCBmZWF0dXJlcyAoZW5zMzsgZm91bmQgd2l0aCBjdHgyL2N0eDMgb24gdGhlIGhvbGRvdXQpLgoKNDAlIG9mIFNvdXJjZSAxIG5hbWVzIG9jY3VyIG1vcmUgdGhhbiBvbmNlIChvbmUgYnVzaW5lc3MgbmFtZSwgc2V2ZXJhbCBhZGRyZXNzZXMpLCBhbmQgbWFueQpTb3VyY2UgMSBhZGRyZXNzZXMgaG9sZCBzZXZlcmFsIGJ1c2luZXNzZXMuIEEgU291cmNlIDIvMyBjb3B5IHdpdGggYW4gZW1wdHkgYWRkcmVzcyAob3Igd2l0aCBhCm1hZGUtdXAgbmFtZSkgZml0cyBldmVyeSBzdWNoIHJpdmFsIGFib3V0IGVxdWFsbHksIHNvIGEgcGFpciBtb2RlbCB0aGF0IGxvb2tzIGF0IE9ORSBwYWlyIGdpdmVzCmVhY2ggfjEvbi4gVGhlIGNvcHkgd2FzIG1hZGUgZnJvbSBPTkUgb2YgdGhlbTogaXRzIGV4YWN0IHNwZWxsaW5nICgiTHRkIiB2cyAiTGltaXRlZCIsICJJbmMiIG9yCm5vdCwgdGhlIG51bWJlcnMgb2YgdGhlIGFkZHJlc3MpIHBvaW50cyBhdCB0aGUgb3duZXIuIEZvciBldmVyeSB1bnN1cmUgcGFpciB0aGVzZSBmZWF0dXJlcyBjb21wYXJlCnRoZSBjb3B5IHdpdGggQUxMIFNvdXJjZSAxIHJlY29yZHMgb2YgdGhlIHNwbGl0IHRoYXQgc2hhcmUgaXRzIG5hbWUga2V5IChvciBpdHMgYWRkcmVzcyBudW1iZXJzKS4KVGhleSBhcmUgY291bnRlZCBvdmVyIHRoZSBGVUxMIFNvdXJjZSAxIHRhYmxlIG9mIHRoZSBzcGxpdCAodHJhaW4gb3IgdGVzdCksIHNvIHRoZSAyNSUgdHJhaW5pbmcKc2FtcGxlIGRvZXMgbm90IGNoYW5nZSB0aGVtOiB0cmFpbiBhbmQgdGVzdCBnZXQgdGhlIHNhbWUga2luZCBvZiBudW1iZXJzLgpPbmx5IHBhaXJzIHdpdGggYSBzdGFnZS0xIHByb2JhYmlsaXR5IGluIChMT1csIEhJR0gpIGdldCB0aGVtICh0aGUgb3RoZXJzOiBOYU4pLCBpbiB0cmFpbiBhbmQgdGVzdCBhbGlrZS4KIiIiCmltcG9ydCBvcwpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBudW1weSBhcyBucAppbXBvcnQgcGFuZGFzIGFzIHBkCmltcG9ydCBweWFycm93IGFzIHBhCmltcG9ydCBweWFycm93LmNzdiBhcyBwYwpmcm9tIHJhcGlkZnV6eiBpbXBvcnQgZnV6egoKTE9XLCBISUdIID0gMC4wMDUsIDAuOTk1Ck1BWF9SSVZBTFMgPSAzMDAKTEVHQUwgPSBzZXQoImxsYyBpbmMgbHRkIGxpbWl0ZWQgcHZ0IHByaXZhdGUgY29ycCBjb3Jwb3JhdGlvbiBjbyBjb21wYW55IHRoZSBscCBsbHAgcGMgcGEgcGxjIHBsbGMgc2FybCBzYXMgIgogICAgICAgICAgICAic2FzdSBldXJsIHNhIHNjaSBhbmQgb2YiLnNwbGl0KCkpClJJVkFMX0NPTFVNTlMgPSBbInJ2X2VtcHR5IiwgInJ2X2tleV9lcSIsICJydl9uMSIsICJydl9uMiIsICJydl9zaW0iLCAicnZfdHNldCIsICJydl9nYXBfYmVzdCIsICJydl9uX2JldHRlciIsCiAgICAgICAgICAgICAgICAgInJ2X25fdGllIiwgInJ2X2dhcF9zZWNvbmQiLCAicnZfYXNpbSIsICJydl9hZF9uMiIsICJydl9hZF9nYXBfYmVzdCIsICJydl9hZF9iZXR0ZXIiLAogICAgICAgICAgICAgICAgICJydl9hZF9nYXBfc2Vjb25kIiwgInJ2X25fd29yZHMiLCAicnZfbWF4X3dvcmQiXQpfVEFCTEVTID0ge30KCgpkZWYgX3JlYWQocGF0aCwgY29scyk6CiAgICB0ID0gcGMucmVhZF9jc3YocGF0aCwgcGFyc2Vfb3B0aW9ucz1wYy5QYXJzZU9wdGlvbnMoZGVsaW1pdGVyPSJcdCIsIHF1b3RlX2NoYXI9RmFsc2UpLAogICAgICAgICAgICAgICAgICAgIGNvbnZlcnRfb3B0aW9ucz1wYy5Db252ZXJ0T3B0aW9ucyhpbmNsdWRlX2NvbHVtbnM9Y29scywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgY29sdW1uX3R5cGVzPXtjOiBwYS5zdHJpbmcoKSBmb3IgYyBpbiBjb2xzfSkpCiAgICByZXR1cm4ge2M6IHQuY29sdW1uKGMpLmNvbWJpbmVfY2h1bmtzKCkgZm9yIGMgaW4gY29sc30KCgpkZWYgcGxhaW4odGV4dHMpOgogICAgcyA9IHBkLlNlcmllcyh0ZXh0cywgZHR5cGU9b2JqZWN0KS5maWxsbmEoIiIpLnN0ci5sb3dlcigpCiAgICBzID0gcy5zdHIubm9ybWFsaXplKCJORktEIikuc3RyLmVuY29kZSgiYXNjaWkiLCAiaWdub3JlIikuc3RyLmRlY29kZSgiYXNjaWkiKQogICAgcmV0dXJuIHMuc3RyLnJlcGxhY2UociJccysiLCAiICIsIHJlZ2V4PVRydWUpLnN0ci5zdHJpcCgpCgoKZGVmIG5hbWVfa2V5KHBsKToKICAgIHMgPSBwbC5zdHIucmVwbGFjZShyInd3d1wuXFMrfGh0dHBzPzovL1xTK3xcUytAXFMrfFxTK1wuY29tXGIiLCAiICIsIHJlZ2V4PVRydWUpCiAgICBzID0gcy5zdHIucmVwbGFjZShyIlwoaWQ6P1xzKlxkK1wpfFxkezYsfSIsICIgIiwgcmVnZXg9VHJ1ZSkKICAgIHMgPSBzLnN0ci5yZXBsYWNlKHIiW15hLXowLTkgXSIsICIgIiwgcmVnZXg9VHJ1ZSkKICAgIHMgPSBzLnN0ci5yZXBsYWNlKHIiXGIoXHcrKSggXDFcYikrIiwgciJcMSIsIHJlZ2V4PVRydWUpCiAgICByZXR1cm4gcy5tYXAobGFtYmRhIHg6ICIgIi5qb2luKHcgZm9yIHcgaW4geC5zcGxpdCgpIGlmIHcgbm90IGluIExFR0FMKSkKCgpkZWYgZGlnaXRfa2V5KGFkZHIpOgogICAgcmV0dXJuIGFkZHIuc3RyLmZpbmRhbGwociJcZCsiKS5tYXAobGFtYmRhIHg6ICIgIi5qb2luKHNvcnRlZChzZXQoeCkpKSkKCgpkZWYgX2dyb3VwcyhrZXlzKToKICAgICIiImtleSAtPiAoc29ydGVkIHJvdyBudW1iZXJzLCBib3VuZHMpIHdpdGhvdXQgYSBQeXRob24gZGljdCBvZiBhcnJheXMuIiIiCiAgICBjb2RlcywgdW5pcXVlcyA9IHBkLmZhY3Rvcml6ZShrZXlzKQogICAgb3JkZXIgPSBucC5hcmdzb3J0KGNvZGVzLCBraW5kPSJzdGFibGUiKQogICAgYm91bmRzID0gbnAuc2VhcmNoc29ydGVkKGNvZGVzW29yZGVyXSwgbnAuYXJhbmdlKGxlbih1bmlxdWVzKSArIDEpKQogICAgcmV0dXJuIHBkLkluZGV4KHVuaXF1ZXMpLCBvcmRlciwgYm91bmRzCgoKZGVmIF90YWJsZXMobl9zMjMpOgogICAgIiIiRnVsbCBTb3VyY2UgMSAobmFtZXMsIGtleXMsIGFkZHJlc3NlcykgYW5kIFNvdXJjZSAyLzMgKG5hbWVzLCBhZGRyZXNzZXMpIG9mIHRoZSBzcGxpdCB3aG9zZQogICAgU291cmNlIDIvMyB0YWJsZSBoYXMgbl9zMjMgcm93cy4iIiIKICAgIGlmIG5fczIzIGluIF9UQUJMRVM6CiAgICAgICAgcmV0dXJuIF9UQUJMRVNbbl9zMjNdCiAgICBkYXRhID0gUGF0aChvcy5lbnZpcm9uWyJBTUwyNl9EQVRBIl0pCiAgICBmb3Igc3BsaXQgaW4gKCJ0ZXN0IiwgInRyYWluIik6CiAgICAgICAgZCA9IGRhdGEgLyBzcGxpdAogICAgICAgIGlmIG5vdCAoZCAvIGYie3NwbGl0fV9zb3VyY2UxLnRzdiIpLmV4aXN0cygpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIHMyMyA9IFtfcmVhZChkIC8gZiJ7c3BsaXR9X3NvdXJjZXtrfS50c3YiLCBbImJ1c2luZXNzX25hbWUiLCAiYnVzaW5lc3NfYWRkcmVzcyJdKSBmb3IgayBpbiAoMiwgMyldCiAgICAgICAgaWYgc3VtKGxlbih4WyJidXNpbmVzc19uYW1lIl0pIGZvciB4IGluIHMyMykgIT0gbl9zMjM6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgczEgPSBfcmVhZChkIC8gZiJ7c3BsaXR9X3NvdXJjZTEudHN2IiwgWyJidXNpbmVzc19uYW1lIiwgImJ1c2luZXNzX2FkZHJlc3MiXSkKICAgICAgICB0ID0geyJzMV9wbGFpbiI6IHBsYWluKHMxWyJidXNpbmVzc19uYW1lIl0udG9fbnVtcHkoemVyb19jb3B5X29ubHk9RmFsc2UpKSwKICAgICAgICAgICAgICJzMV9hZGRyIjogcGxhaW4oczFbImJ1c2luZXNzX2FkZHJlc3MiXS50b19udW1weSh6ZXJvX2NvcHlfb25seT1GYWxzZSkpLAogICAgICAgICAgICAgInMyM19uYW1lIjogcGEuY29uY2F0X2FycmF5cyhbeFsiYnVzaW5lc3NfbmFtZSJdIGZvciB4IGluIHMyM10pLAogICAgICAgICAgICAgInMyM19hZGRyIjogcGEuY29uY2F0X2FycmF5cyhbeFsiYnVzaW5lc3NfYWRkcmVzcyJdIGZvciB4IGluIHMyM10pfQogICAgICAgIGRlbCBzMSwgczIzCiAgICAgICAgdFsiczFfa2V5Il0gPSBuYW1lX2tleSh0WyJzMV9wbGFpbiJdKQogICAgICAgIHRbInMxX2RrZXkiXSA9IGRpZ2l0X2tleSh0WyJzMV9hZGRyIl0pCiAgICAgICAgdFsibmFtZV9ncm91cHMiXSA9IF9ncm91cHModFsiczFfa2V5Il0udG9fbnVtcHkoKSkKICAgICAgICB0WyJhZGRyX2dyb3VwcyJdID0gX2dyb3Vwcyh0WyJzMV9ka2V5Il0udG9fbnVtcHkoKSkKICAgICAgICB0WyJuX2tleSJdID0gdFsiczFfa2V5Il0ubWFwKHRbInMxX2tleSJdLnZhbHVlX2NvdW50cygpKS50b19udW1weShkdHlwZT1ucC5mbG9hdDMyKQogICAgICAgIHByaW50KGYicml2YWw6IHtzcGxpdH0gdGFibGVzIGxvYWRlZCAoe2xlbih0WydzMV9wbGFpbiddKTosfSBTb3VyY2UgMSwge25fczIzOix9IFNvdXJjZSAyLzMgcmVjb3JkcykiLCBmbHVzaD1UcnVlKQogICAgICAgIF9UQUJMRVNbbl9zMjNdID0gdAogICAgICAgIHJldHVybiB0CiAgICByYWlzZSBTeXN0ZW1FeGl0KGYicml2YWw6IG5vIHNwbGl0IHdpdGgge25fczIzOix9IFNvdXJjZSAyLzMgcmVjb3JkcyB1bmRlciB7ZGF0YX0iKQoKCmRlZiByZWxlYXNlKCk6CiAgICBfVEFCTEVTLmNsZWFyKCkKCgpkZWYgX3JpdmFscyhxdWVyeSwgb3duLCBncm91cHMsIGtleSwgczFfdGV4dHMsIHNjb3Jlciwgb3duX3Njb3JlKToKICAgICIiIlNjb3JlcyBvZiB0aGUgcXVlcnkgdGV4dCBhZ2FpbnN0IGV2ZXJ5IFNvdXJjZSAxIHJlY29yZCB3aXRoIHRoaXMga2V5LiIiIgogICAgaW5kZXgsIG9yZGVyLCBib3VuZHMgPSBncm91cHMKICAgIGNvZGUgPSBpbmRleC5nZXRfaW5kZXhlcihba2V5XSlbMF0KICAgIGlmIGNvZGUgPCAwOgogICAgICAgIHJldHVybiBOb25lCiAgICBycyA9IG9yZGVyW2JvdW5kc1tjb2RlXTpib3VuZHNbY29kZSArIDFdXQogICAgaWYgbGVuKHJzKSA9PSAwIG9yIGxlbihycykgPiBNQVhfUklWQUxTOgogICAgICAgIHJldHVybiBOb25lCiAgICBzYyA9IG5wLmZyb21pdGVyKChzY29yZXIocXVlcnksIHMxX3RleHRzW3JdKSBmb3IgciBpbiBycyksIGR0eXBlPW5wLmZsb2F0MzIsIGNvdW50PWxlbihycykpCiAgICBvdGhlcnMgPSBzY1tycyAhPSBvd25dCiAgICByZXR1cm4gc2MubWF4KCksIChzYyA+IG93bl9zY29yZSkuc3VtKCksIChzYyA9PSBvd25fc2NvcmUpLnN1bSgpIC0gKG93biBpbiBycyksIChvdGhlcnMubWF4KCkgaWYgbGVuKG90aGVycykgZWxzZSBucC5uYW4pCgoKZGVmIHJpdmFsX2ZlYXR1cmVzKHMxX3JvdywgczIzX3JvdywgcCwgbl9zMjMpOgogICAgIiIiZGljdCBSSVZBTF9DT0xVTU5TIC0+IGZsb2F0MzIgYXJyYXlzIChOYU4gb3V0c2lkZSB0aGUgdW5zdXJlIHdpbmRvdykuIiIiCiAgICBuID0gbGVuKHApCiAgICBvdXQgPSB7YzogbnAuZnVsbChuLCBucC5uYW4sIGR0eXBlPW5wLmZsb2F0MzIpIGZvciBjIGluIFJJVkFMX0NPTFVNTlN9CiAgICB3ID0gbnAuZmxhdG5vbnplcm8oKHAgPiBMT1cpICYgKHAgPCBISUdIKSkKICAgIGlmIGxlbih3KSA9PSAwOgogICAgICAgIHJldHVybiBvdXQKICAgIHQgPSBfdGFibGVzKG5fczIzKQogICAgYSA9IHMxX3Jvd1t3XQogICAgdSwgaW52ID0gbnAudW5pcXVlKHMyM19yb3dbd10sIHJldHVybl9pbnZlcnNlPVRydWUpCiAgICB0YWtlID0gcGEuYXJyYXkodSkKICAgIHJfcGxhaW4gPSBwbGFpbih0WyJzMjNfbmFtZSJdLnRha2UodGFrZSkudG9fbnVtcHkoemVyb19jb3B5X29ubHk9RmFsc2UpKQogICAgcl9hZGRyID0gcGxhaW4odFsiczIzX2FkZHIiXS50YWtlKHRha2UpLnRvX251bXB5KHplcm9fY29weV9vbmx5PUZhbHNlKSkKICAgIHJfa2V5LCByX2RrZXkgPSBuYW1lX2tleShyX3BsYWluKSwgZGlnaXRfa2V5KHJfYWRkcikKICAgIGJfcGxhaW4sIGJfa2V5ID0gcl9wbGFpbi50b19udW1weSgpW2ludl0sIHJfa2V5LnRvX251bXB5KClbaW52XQogICAgYl9hZGRyLCBiX2RrZXkgPSByX2FkZHIudG9fbnVtcHkoKVtpbnZdLCByX2RrZXkudG9fbnVtcHkoKVtpbnZdCiAgICBzMV9wbGFpbiwgczFfa2V5LCBzMV9hZGRyID0gdFsiczFfcGxhaW4iXS50b19udW1weSgpLCB0WyJzMV9rZXkiXS50b19udW1weSgpLCB0WyJzMV9hZGRyIl0udG9fbnVtcHkoKQogICAgczFfZGtleSA9IHRbInMxX2RrZXkiXS50b19udW1weSgpCiAgICBhX3BsYWluLCBhX2tleSwgYV9hZGRyID0gczFfcGxhaW5bYV0sIHMxX2tleVthXSwgczFfYWRkclthXQogICAgZW1wdHkgPSAoYl9hZGRyID09ICIiKS5hc3R5cGUobnAuZmxvYXQzMikKICAgIHNpbSA9IG5wLmFycmF5KFtmdXp6LnJhdGlvKHgsIHkpIGZvciB4LCB5IGluIHppcChhX3BsYWluLCBiX3BsYWluKV0sIGR0eXBlPW5wLmZsb2F0MzIpCiAgICBhc2ltID0gbnAuYXJyYXkoW2Z1enoudG9rZW5fc2V0X3JhdGlvKHgsIHkpIGZvciB4LCB5IGluIHppcChhX2FkZHIsIGJfYWRkcildLCBkdHlwZT1ucC5mbG9hdDMyKQogICAga2MgPSB0WyJzMV9rZXkiXS52YWx1ZV9jb3VudHMoKQogICAgbjIgPSBwZC5TZXJpZXMoYl9rZXkpLm1hcChrYykuZmlsbG5hKDApLnRvX251bXB5KGR0eXBlPW5wLmZsb2F0MzIpCiAgICBkYyA9IHRbInMxX2RrZXkiXS52YWx1ZV9jb3VudHMoKQogICAgYWRfbjIgPSBwZC5TZXJpZXMoYl9ka2V5KS5tYXAoZGMpLmZpbGxuYSgwKS50b19udW1weShkdHlwZT1ucC5mbG9hdDMyKQogICAgYWRfbjJbYl9ka2V5ID09ICIiXSA9IC0xCiAgICBjb2xzID0ge2M6IG5wLmZ1bGwobGVuKHcpLCBucC5uYW4sIGR0eXBlPW5wLmZsb2F0MzIpIGZvciBjIGluIFJJVkFMX0NPTFVNTlN9CiAgICBmb3IgaSBpbiByYW5nZShsZW4odykpOgogICAgICAgIHIgPSBfcml2YWxzKGJfcGxhaW5baV0sIGFbaV0sIHRbIm5hbWVfZ3JvdXBzIl0sIGJfa2V5W2ldLCBzMV9wbGFpbiwgZnV6ei5yYXRpbywgc2ltW2ldKQogICAgICAgIGlmIHIgaXMgbm90IE5vbmU6CiAgICAgICAgICAgIGNvbHNbInJ2X2dhcF9iZXN0Il1baV0sIGNvbHNbInJ2X25fYmV0dGVyIl1baV0sIGNvbHNbInJ2X25fdGllIl1baV0gPSBzaW1baV0gLSByWzBdLCByWzFdLCByWzJdCiAgICAgICAgICAgIGNvbHNbInJ2X2dhcF9zZWNvbmQiXVtpXSA9IHNpbVtpXSAtIHJbM10KICAgICAgICBpZiBiX2RrZXlbaV06CiAgICAgICAgICAgIHIgPSBfcml2YWxzKGJfYWRkcltpXSwgYVtpXSwgdFsiYWRkcl9ncm91cHMiXSwgYl9ka2V5W2ldLCBzMV9hZGRyLCBmdXp6LnRva2VuX3NldF9yYXRpbywgYXNpbVtpXSkKICAgICAgICAgICAgaWYgciBpcyBub3QgTm9uZToKICAgICAgICAgICAgICAgIGNvbHNbInJ2X2FkX2dhcF9iZXN0Il1baV0sIGNvbHNbInJ2X2FkX2JldHRlciJdW2ldID0gYXNpbVtpXSAtIHJbMF0sIHJbMV0KICAgICAgICAgICAgICAgIGNvbHNbInJ2X2FkX2dhcF9zZWNvbmQiXVtpXSA9IGFzaW1baV0gLSByWzNdCiAgICB3b3JkcyA9IHBkLlNlcmllcyhiX3BsYWluKS5zdHIuc3BsaXQoKQogICAgY29scy51cGRhdGUoewogICAgICAgICJydl9lbXB0eSI6IGVtcHR5LCAicnZfa2V5X2VxIjogKGFfa2V5ID09IGJfa2V5KS5hc3R5cGUobnAuZmxvYXQzMiksICJydl9uMSI6IHRbIm5fa2V5Il1bYV0sCiAgICAgICAgInJ2X24yIjogbjIsICJydl9zaW0iOiBzaW0sCiAgICAgICAgInJ2X3RzZXQiOiBucC5hcnJheShbZnV6ei50b2tlbl9zZXRfcmF0aW8oeCwgeSkgZm9yIHgsIHkgaW4gemlwKGFfa2V5LCBiX2tleSldLCBkdHlwZT1ucC5mbG9hdDMyKSwKICAgICAgICAicnZfYXNpbSI6IGFzaW0sICJydl9hZF9uMiI6IGFkX24yLAogICAgICAgICJydl9uX3dvcmRzIjogd29yZHMuc3RyLmxlbigpLnRvX251bXB5KGR0eXBlPW5wLmZsb2F0MzIpLAogICAgICAgICJydl9tYXhfd29yZCI6IHdvcmRzLm1hcChsYW1iZGEgeDogbWF4KChsZW4oeSkgZm9yIHkgaW4geCksIGRlZmF1bHQ9MCkpLnRvX251bXB5KGR0eXBlPW5wLmZsb2F0MzIpfSkKICAgIGZvciBjIGluIFJJVkFMX0NPTFVNTlM6CiAgICAgICAgb3V0W2NdW3ddID0gY29sc1tjXQogICAgcHJpbnQoZiJyaXZhbDogZmVhdHVyZXMgZm9yIHtsZW4odyk6LH0gb2Yge246LH0gcGFpcnMiLCBmbHVzaD1UcnVlKQogICAgcmV0dXJuIG91dAo="
POS, NEG = 0.95, 0.02
PSEUDO_SHARE = 0.12        # pseudo pairs at most 12% of the real learning pairs (0.30 ran out of memory: 31 GB)
EXTRA_THRESHOLDS = (0.5, 0.6, 0.8)
N_JOBS = 4
PACKAGES = ["anyascii==0.3.3", "rapidfuzz==3.14.6", "sparse_dot_topn==1.2.0"]
PIPELINE_TIME_LIMIT = 11.25 * 3600

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA = WORKING / "data"
WORK = WORKING / "work"
SRC_MODELS = WORKING / "src_models"


def say(message):
    print(f"### {message}", flush=True)


def find(name):
    return sorted(p for root in (INPUT, WORKING / "unzipped") if root.exists() for p in root.rglob(name))


def link(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.symlink(source, target)


def combine_ce_scores():
    """The ensemble: for every pair, the mean of the cross-encoder scores (logits) of all
    LLM-layer notebooks given as input (aml26-llm<N>). Written to WORK/<split>/ce_scores.parquet
    BEFORE the other inputs are linked, so train / assign use the mean instead of one model."""
    import re
    import pandas as pd
    for split in ("train", "test"):
        files = sorted(p for p in INPUT.rglob(f"work/{split}/ce_scores.parquet")
                       if re.search(r"aml26-llm\d", str(p)))
        if not files:
            raise SystemExit(f"no {split} ce_scores.parquet of an aml26-llm notebook in the inputs")
        parts = []
        for i, path in enumerate(files):
            part = pd.read_parquet(path, columns=["s1_row", "s23_row", "ce"])
            say(f"{split}: {path} -> {len(part):,} pairs, ce mean {part['ce'].mean():.3f} sd {part['ce'].std():.3f}")
            parts.append(part.rename(columns={"ce": f"ce{i}"}).set_index(["s1_row", "s23_row"]))
        wide = pd.concat(parts, axis=1, join="outer")
        common = wide.dropna()
        say(f"{split}: {len(common):,} pairs scored by all {len(files)} models; correlations:\n{common.corr().round(4)}")
        out = wide.mean(axis=1).rename("ce").astype("float32").reset_index()
        target = WORK / split / "ce_scores.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)
        say(f"{split}: mean score of {len(out):,} pairs -> {target}")


def prepare():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *PACKAGES])
    if not find("test_source1.tsv"):
        for archive in find("*.zip"):
            zipfile.ZipFile(archive).extractall(WORKING / "unzipped")
    for split, names in {"train": ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv",
                                   "train_ground_truth.tsv"],
                         "test": ["test_source1.tsv", "test_source2.tsv", "test_source3.tsv"]}.items():
        for name in names:
            link(find(name)[0], DATA / split / name)
    # SRC's cross-encoder scores (what its model was trained with): gen uses them to reproduce SRC's p
    found = [p for p in INPUT.rglob("work/test/ce_scores.parquet") if SRC in str(p)]
    if not found:
        raise SystemExit(f"{SRC} work/test/ce_scores.parquet not found (did {SRC} finish?)")
    link(found[0], WORK / "test" / "ce_src.parquet")
    say(f"test/ce_src.parquet <- {found[0]}")
    combine_ce_scores()      # the new mean of all llm notebooks -> work/<split>/ce_scores.parquet
    meta = [p for p in INPUT.rglob("work/models/meta.json") if SRC in str(p)]
    if not meta:
        raise SystemExit(f"{SRC} models not found")
    for path in meta[0].parent.iterdir():
        link(path, SRC_MODELS / path.name)
    # everything else (features, cleaned tables, candidates): SRC's files first (its 30-candidate
    # features win), then the block notebooks; the llm notebooks' scores are already in the mean
    import re
    count = 0
    paths = [p for p in INPUT.rglob("work/*/*") if p.is_file() and p.parent.name in ("train", "test")
             and not re.search(r"aml26-llm\d", str(p))]
    for path in sorted(paths, key=lambda p: SRC not in str(p)):
        if not (WORK / path.parent.name / path.name).exists():
            link(path, WORK / path.parent.name / path.name)
            count += 1
    say(f"reusing {count} files of the input notebooks")
    for needed in ["train/features.parquet", "train/candidates.parquet", "test/features.parquet",
                   "test/candidate_pairs.tsv", "test/s1_clean.parquet", "test/s23_clean.parquet"]:
        if not (WORK / needed).exists():
            raise SystemExit(f"work/{needed} missing")


def code_dir():
    return find("run_pipeline.py")[0].parent


# ---------------------------------------------------------------------------
# gen: the pseudo-labelled test pairs (runs in its own process: frees its memory)
# ---------------------------------------------------------------------------
def gen():
    import numpy as np
    import pandas as pd
    import lightgbm as lgb
    import pyarrow.parquet as pq
    sys.path.insert(0, "/tmp/aml26_code")
    from stage2 import collective_features
    from rival import RIVAL_COLUMNS
    from cross_encoder import ce_features

    split_dir = WORK / "test"
    meta = json.loads((SRC_MODELS / "meta.json").read_text())
    model = lgb.Booster(model_file=str(SRC_MODELS / "lgbm.txt"))
    stage1 = [lgb.Booster(model_file=str(SRC_MODELS / n)) for n in meta["stage1_models"]]
    first = meta["stage1_features"]
    file = pq.ParquetFile(split_dir / "features.parquet")

    def batches(columns):
        for batch in file.iter_batches(batch_size=1_000_000, columns=["s1_row", "s23_row"] + columns):
            yield batch.to_pandas()

    rows, p1 = [], []
    for batch in batches(first):
        rows.append(batch[["s1_row", "s23_row"]].to_numpy())
        p1.append((sum(m.predict(batch[first]) for m in stage1) / len(stage1)).astype(np.float32))
    rows = np.concatenate(rows)
    p1 = np.concatenate(p1)
    s1_row, s23_row = rows[:, 0], rows[:, 1]
    del rows
    say(f"gen: stage 1 of {len(p1):,} test pairs done")
    s23 = pd.read_parquet(split_dir / "s23_clean.parquet", columns=["name_core", "addr_clean"])
    extra = collective_features([(s1_row, s23_row, p1)], s23["name_core"].to_numpy(),
                                s23["addr_clean"].to_numpy())[0]
    del s23
    collective = extra
    scores = pd.read_parquet(split_dir / "ce_src.parquet")
    extra = pd.concat([collective, ce_features(s1_row, s23_row, scores)], axis=1)
    del scores
    say(f"gen: collective + cross-encoder features ready {list(extra.columns)}")

    p = np.empty(len(p1), dtype=np.float32)
    start = 0
    for batch in batches(first):
        stop = start + len(batch)
        batch = pd.concat([batch.reset_index(drop=True), extra.iloc[start:stop].reset_index(drop=True)], axis=1)
        p[start:stop] = model.predict(batch[meta["features"]])
        start = stop
    saved = [q for q in INPUT.rglob("test_pair_probabilities.parquet") if SRC in str(q)]
    if saved:
        old = pd.read_parquet(saved[0])["probability"].to_numpy()
        if len(old) == len(p):
            say(f"gen: check vs {SRC}'s saved probabilities: max abs diff {np.abs(old - p).max():.2e}")

    del extra
    gc.collect()
    scores = pd.read_parquet(split_dir / "ce_scores.parquet")
    new_ce = ce_features(s1_row, s23_row, scores)
    del scores
    say(f"gen: new cross-encoder mean: {new_ce.notna().mean().round(3).to_dict()} of pairs scored")
    extra = pd.concat([collective, new_ce], axis=1)
    del collective, new_ce

    # exclusivity winners: the best s1 for every s23
    order = np.lexsort((-p, s23_row))
    winner = np.zeros(len(p), dtype=bool)
    first_of = np.r_[True, s23_row[order][1:] != s23_row[order][:-1]]
    winner[order[first_of]] = True
    positive = winner & (p >= POS)
    negative = p <= NEG

    labels = pq.read_table(WORK / "train" / "features.parquet", columns=["label"]).column(0).to_numpy()
    n_learn = int(len(labels) * 0.8)
    rate = float(labels.mean())
    del labels
    cap_pos, cap_neg = int(PSEUDO_SHARE * n_learn * rate), int(PSEUDO_SHARE * n_learn * (1 - rate))
    rng = np.random.default_rng(7)

    def sample(mask, cap):
        idx = np.flatnonzero(mask)
        return np.sort(rng.choice(idx, cap, replace=False)) if len(idx) > cap else idx

    pos_idx, neg_idx = sample(positive, cap_pos), sample(negative, cap_neg)
    say(f"gen: sure test pairs: {positive.sum():,} positive (p>={POS}, exclusive), {negative.sum():,} negative "
        f"(p<={NEG}); train positive rate {rate:.3f}, learning pairs ~{n_learn:,}; "
        f"taking {len(pos_idx):,} + {len(neg_idx):,}")
    country = pd.read_parquet(split_dir / "s1_clean.parquet", columns=["country"])["country"].to_numpy()
    say("gen: positives by country (log only) " + str(pd.Series(country[s1_row[pos_idx]]).value_counts().to_dict()))
    say("gen: test records by country (log only) " + str(pd.Series(country).value_counts().to_dict()))
    say("gen: p between 0.3 and 0.9 (unsure) by country (log only) "
        + str(pd.Series(country[s1_row[winner & (p > 0.3) & (p < 0.9)]]).value_counts().to_dict()))

    take = np.zeros(len(p), dtype=bool)
    take[pos_idx] = True
    take[neg_idx] = True
    label = positive.astype(np.int8)
    parts = []
    start = 0
    for batch in batches(first):
        stop = start + len(batch)
        sel = take[start:stop]
        if sel.any():
            part = pd.concat([batch.reset_index(drop=True), extra.iloc[start:stop].reset_index(drop=True)], axis=1)[sel]
            part = part[list(dict.fromkeys(meta["features"] + RIVAL_COLUMNS))].copy()
            part["label"] = label[start:stop][sel]
            parts.append(part)
        start = stop
    out = pd.concat(parts, ignore_index=True)
    (WORK / "pseudo").mkdir(parents=True, exist_ok=True)
    out.to_parquet(WORK / "pseudo" / "pseudo_pairs.parquet", index=False)
    say(f"gen: wrote {len(out):,} pseudo pairs ({out['label'].mean():.3f} positive), {out.shape[1]} columns")


# ---------------------------------------------------------------------------
# patched copy of the code
# ---------------------------------------------------------------------------
TRAIN_OLD = "    model, curve = train_model(learning, stopping, columns)\n"
TRAIN_NEW = """    pseudo_file = work_dir / "pseudo" / "pseudo_pairs.parquet"
    if pseudo_file.exists():
        pseudo = pd.read_parquet(pseudo_file)
        missing = [c for c in columns if c not in pseudo.columns]
        if missing:
            raise SystemExit(f"pseudo pairs lack the columns {missing}")
        n_real = len(learning)
        learning = learning[columns + ["label"]]
        pseudo = pseudo[columns + ["label"]]
        for c in columns:       # column by column: float32 halves the frames without a full copy
            if learning[c].dtype == np.float64:
                learning[c] = learning[c].astype(np.float32)
            if pseudo[c].dtype != learning[c].dtype:
                pseudo[c] = pseudo[c].astype(learning[c].dtype)
        gc.collect()
        learning = pd.concat([learning, pseudo], ignore_index=True)
        log(f"PSEUDO: learning = {n_real:,} train pairs + {len(pseudo):,} sure test pairs "
            f"({pseudo['label'].mean():.3f} positive); early stop + holdout stay train only")
        del pseudo
        gc.collect()
""" + TRAIN_OLD

ASSIGN_OLD = """    log(f"wrote {output_dir / 'matching_results.tsv'} and candidate_pairs.tsv")\n"""
ASSIGN_NEW = ASSIGN_OLD + f"""    try:
        extra_dir = output_dir / "thresholds"
        extra_dir.mkdir(parents=True, exist_ok=True)
        pairs[["s1_row", "s23_row", "probability"]].to_parquet(extra_dir / "test_pair_probabilities.parquet", index=False)
        for t in {EXTRA_THRESHOLDS!r}:
            extra = decide(pairs, "threshold", threshold=t)
            write_id_lists(extra_dir / f"matching_results_t{{t}}.tsv", s1_ids, extra, s23_ids, header="matched_entity_ids")
            log(f"threshold {{t}}: {{len(extra):,}} pairs -> thresholds/matching_results_t{{t}}.tsv")
    except Exception as error:
        log(f"extra threshold files failed (main answer is fine): {{error!r}}")
"""


S2_COLS_OLD = """                  "sib_addr_support", "sib_name_support"]
"""
S2_COLS_NEW = """                  "sib_addr_support", "sib_name_support"] + __import__("rival").RIVAL_COLUMNS
"""
S2_OLD = """    frames = []
    for order, _, _, numbers in prepared:
"""
S2_NEW = """    from rival import rival_features, release
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
"""
DEC_OLD = """    result = evaluate(model, columns, holdout, holdout_rows, truth, "HOLDOUT")
"""
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


def patched_code():
    code = Path("/tmp/aml26_code")
    shutil.rmtree(code, ignore_errors=True)
    shutil.copytree(code_dir(), code)
    import base64
    (code / "rival.py").write_bytes(base64.b64decode(RIVAL_B64))
    for name, old, new in (("train.py", TRAIN_OLD, TRAIN_NEW), ("assign.py", ASSIGN_OLD, ASSIGN_NEW),
                           ("stage2.py", S2_COLS_OLD, S2_COLS_NEW), ("stage2.py", S2_OLD, S2_NEW),
                           ("train.py", DEC_OLD, DEC_NEW), ("stage2.py", FR_OLD, FR_NEW),
                           ("assign.py", CE_OLD, CE_NEW)):
        text = (code / name).read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise SystemExit(f"{name} patch failed (code dataset changed?)")
        (code / name).write_text(text.replace(old, new), encoding="utf-8")
    return code


def run(command, label):
    time_left = PIPELINE_TIME_LIMIT - (time.time() - NOTEBOOK_START)
    say(f"{label}: {' '.join(map(str, command))} (time left {time_left / 3600:.2f} h)")
    try:
        returncode = subprocess.run(command, timeout=time_left).returncode
    except subprocess.TimeoutExpired:
        returncode = -1
    import resource
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    say(f"peak memory {peak:.1f} GB; time {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
    if returncode != 0:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "FAILED.txt").write_text(f"{label}: exit code {returncode}\n")
        say(f"PIPELINE FAILED ({label}: exit code {returncode}{' = killed, out of memory?' if returncode == -9 else ''})")
        return False
    say(f"PIPELINE OK ({label})")
    return True


def pipeline(code, *args):
    return run([sys.executable, "-u", str(code / "run_pipeline.py"), *args, "--data-dir", str(DATA),
                "--work-dir", str(WORK), "--n-jobs", str(N_JOBS)], args[0])


def validate():
    output = WORK / "output"
    validator = find("validate_submission.py")
    if validator and (output / "matching_results.tsv").exists():
        subprocess.run([sys.executable, str(validator[0]), "-m", str(output / "matching_results.tsv"),
                        "-c", str(output / "candidate_pairs.tsv"), "-t", str(DATA / "test")])


def clean_up():
    for path in WORKING.rglob("*"):
        if path.is_symlink():
            path.unlink()
    (WORK / "test" / "ce_src.parquet").unlink(missing_ok=True)
    for folder in (DATA, WORKING / "unzipped", SRC_MODELS):
        shutil.rmtree(folder, ignore_errors=True)
    (WORK / "pseudo" / "pseudo_pairs.parquet").unlink(missing_ok=True)   # big; the log has its numbers


def gen_in_child():
    """gen in a forked process (its ~10 GB go back to the system when it ends)."""
    import multiprocessing
    say("gen: pseudo-labelled test pairs")
    child = multiprocessing.get_context("fork").Process(target=gen)
    child.start()
    child.join()
    if child.exitcode != 0:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "FAILED.txt").write_text(f"gen: exit code {child.exitcode}\n")
        say(f"PIPELINE FAILED (gen: exit code {child.exitcode}{' = killed, out of memory?' if child.exitcode == -9 else ''})")
        return False
    say(f"PIPELINE OK (gen); time {(time.time() - NOTEBOOK_START) / 3600:.2f} h")
    return True


def main():
    say(f"aml26-ens3: {NOTE}")
    os.environ["AML26_DATA"] = str(DATA)
    try:
        prepare()
        code = patched_code()
        if (gen_in_child() and pipeline(code, "train", "--note", NOTE) and pipeline(code, "assign")):
            validate()
            say("ALL DONE")
    finally:
        clean_up()


main()

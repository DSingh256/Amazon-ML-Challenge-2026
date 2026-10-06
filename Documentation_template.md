# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** 404 Not found  
**Team Members:** Abhishek Kumar Gupta, Niraj Visave  
**Submission Date:** 27 Sep 2026

---

## 1. Executive Summary

Our pipeline uses only the challenge data. A bidirectional character 3-gram TF-IDF search
proposes 30 Source 2/3 candidates per Source 1 record (pair recall 0.982 on the full training set).
A two-stage LightGBM model then scores every candidate pair. Stage 1 uses about 70 string,
number and retrieval features. Stage 2 adds "collective" features: how the pair compares with
the other candidates of both records, and with same-name or same-address rival Source 1 records.
It also adds the mean score of several fine-tuned multilingual cross-encoders (XLM-RoBERTa
base/large, bge-reranker-v2-m3). Stage 2 is further trained on confidently labelled test pairs
(self-training). The final list per record comes from a one-owner exclusivity rule plus an
F0.5-tuned keep rule. Macro F0.5 on a 66,358-record training holdout rose from 0.957
(first model) to 0.9874.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Size:** train has 2.2M Source 1, 5.0M Source 2 and 5.3M Source 3 records; test has
  1.73M / 4.9M / 5.1M. Brute force (1.7M × 10M) is impossible, so blocking is the first
  bottleneck.
- **Matches per record:** a Source 1 record has 3.46 true matches on average. 5.6% have none,
  and those score 1.0 only with an empty answer, so precision per record matters (F0.5).
- **One owner:** every Source 2/3 record matches at most one Source 1 record. We use this at
  decision time (exclusivity) and in features (competition between Source 1 records).
- **Name noise:**
  - legal-form variants (Pvt Ltd / Private Limited / LLC / L.L.C. / SARL / S.A.R.L.)
  - reordered words ("Ltd. Bangalore Techno")
  - dropped or added words ("Sunrise Private Limited Services")
  - typos and OCR-like swaps (l/I, rn/m)
  - accents
  - names written in Devanagari, Bengali, Kannada and other scripts
- **Address noise:**
  - abbreviations (Rd/Road, St/Street, CT/Court)
  - missing PIN/ZIP, state or the whole address
  - state names vs codes vs native script
  - changed house numbers ("203B" → "2-03B")
  - upper-case copies
  - landmark text
- **Rivals:** 40% of Source 1 names occur more than once (one business, several branches), and
  many addresses host several businesses. A Source 2/3 copy with an empty address fits every
  branch equally. Only its exact spelling, compared against *all* rivals, points to the owner.
- **Unseen country:** the test set contains a country (France) that does not appear in
  training. We never use country as a feature and never tune anything per country.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking, then a two-stage gradient-boosted pair classifier with
collective (graph-like) features and a fine-tuned cross-encoder layer, then constrained
decoding.

**Core Innovations:**
1. **Scalable blocking.** Bidirectional TF-IDF search with rare-3-gram query pruning scales to
   10M records on 4 CPU cores. The reverse direction finds short Source 2/3 records that the
   forward search misses.
2. **Collective stage 2.** Each pair is judged together with its competitors:
   - stage-1 probability rank and the best rival probability on both sides
   - support from sibling copies
   - 17 *rival* features that compare the Source 2/3 name/address with every Source 1 record
     sharing its name key or address numbers
3. **Cross-encoder ensemble.** Five or six fine-tuned multilingual cross-encoders read both raw
   records (any script). Their mean logit is a stage-2 feature, computed only for "unsure" pairs
   (stage-1 p between 0.001 and 0.999).
4. **Self-training.** Test pairs that are confident after stage 1 (exclusive winner with p ≥ 0.95
   as positives, p ≤ 0.02 as negatives; at most 12% of the learning pairs) are added to stage-2
   learning. This adapts the model to the test distribution, including the unseen country,
   without any labels.
5. **Metric-aware decoding.**
   - Exclusivity: each Source 2/3 record goes only to its best Source 1 record.
   - Keep rule: keep a pair when q ≥ t, or when it is the record's top candidate and q ≥ t1
     (1-match records otherwise often get an empty answer).
   - (t, t1) is chosen by macro F0.5 on the holdout with a 2-fold honest check (thresholds
     picked on one half, measured on the other).

**Text cleaning (S1).**
- Every name and address is transliterated to ASCII (`anyascii`), lower-cased and stripped of
  punctuation.
- "M/s" / "Messrs" prefixes are dropped and dotted abbreviations are glued ("l.l.c" → "llc").
- Legal forms are mapped to one standard form and cut from the name (kept as a separate field).
- Address abbreviations are expanded, US state names are mapped to codes, and numbers (house
  numbers, PIN / ZIP / postcodes) are extracted into their own field.
- All rules are generic word lists and none is chosen by country.
- On 20,000 true pairs, the average word Jaccard of the two names rises from 0.455 (raw) to
  0.706 after cleaning.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:** character 3-gram TF-IDF (sublinear tf) over "name + address".
  - The search runs only among records with the same `country` label. This is a generic
    group-by, so an unseen label such as France is handled the same way.
  - Forward search: the top 15 Source 2 and top 15 Source 3 records per Source 1 record.
  - Reverse search: every Source 2/3 record that ranks the Source 1 record in its own top 3.
  - The hits are merged and the best 30 are kept.
  - Each query uses only its 30 rarest 3-grams, and 3-grams in more than 0.1% of documents are
    skipped. The sparse top-k product runs multi-threaded (`sparse_dot_topn`).
- **Candidate pairs generated:** 30 per Source 1 record, so **51,976,320 test pairs**
  (66.2M on train; reduction ratio 0.999994 of all same-country pairs).
- **How we ensured true matches were not lost:** pair recall was measured on *all* 2.2M
  training Source 1 records after every change (`train/blocking_report.txt`).

  Final pair recall is **0.982** (India 0.971, US 0.989), and 94.2% of records have all their
  matches found. The ceiling for a perfect classifier on these candidates is macro F0.5 0.9941.

  | K | 1 | 3 | 5 | 10 | 15 | 20 | 30 | 50 |
  |---|---|---|---|---|---|---|---|---|
  | recall | 0.251 | 0.653 | 0.881 | 0.972 | 0.977 | 0.979 | 0.982 | 0.982 |

  - The reverse search alone finds 312k true pairs that the forward search misses.
  - Going from 15 to 30 candidates raised the holdout F0.5 from 0.9839 to 0.9849.
  - Training uses the same blocking as test, so the retrieval and rank features look the same
    in both.

---

## 4. Matching Model

**Features used:**

- **Stage 1 (about 70 features):**
  - *Name:* Jaro-Winkler, Levenshtein, ratio, partial ratio, token-sort and token-set ratios
    (`rapidfuzz`); word and char TF-IDF cosine; IDF-weighted word Jaccard; first and last word
    equal; exact equality; length and word-count differences; learned name-word weights.
  - *Address:* word and char TF-IDF cosine, IDF-weighted Jaccard, token-set, token-sort and
    partial ratios, missing-address flags.
  - *Numbers:* Jaccard of extracted numbers, number conflict, first number (house number) equal,
    postcode equal. The postcode is the last 4–6 digit number, found by pattern for PIN, ZIP and
    French codes alike.
  - *Legal form:* equal, conflicting or missing.
  - *Retrieval and competition:* search scores and ranks in both directions; gap to the best
    candidate of the same Source 1 record and of the same Source 2/3 record; number of Source 1
    records that want the same Source 2/3 record; source (2 or 3); candidate list size; count of
    same-name siblings in the candidate list.
- **Stage 2 (collective):**
  - stage-1 probability `p1`, its rank in the record, and the max and sum of the other
    candidates' p1
  - number of sure other owners
  - sibling address and name support
  - cross-encoder mean `ce` and `ce_others_max`
  - rival features (unsure pairs only): the Source 2/3 name and address compared with every
    Source 1 record with the same name key or the same address numbers, giving the gap to the
    best rival, the number of better or tied rivals, and the gap to the second best
- **Cross-encoder layer:**
  - Models (all fine-tuned here on training pairs, cross-fitted by record so that no training
    pair is scored by a model that learned from its record):
    - XLM-RoBERTa base: three runs with different training sizes (150k–300k records, hardest 6
      candidates each), plus one self-trained run
    - XLM-RoBERTa large
    - BAAI/bge-reranker-v2-m3
  - Each model reads `name | address` of both records in their original script.
  - AUC on unseen records: 0.9995–0.9996. On unsure training pairs, cross-encoder AUC is 0.989
    vs 0.986 for stage 1.
- `country` is **never** a feature.

**Model type:** LightGBM binary classifiers (stage 1 and stage 2; 127 leaves, learning rate 0.05,
early stopping on an id-hash split).
- Stage 1 is cross-fitted, so the p1 on training pairs comes from a model that did not see
  their record.
- Stage 2 learns from the training pairs plus the pseudo-labelled test pairs.

**Threshold selection method:** macro F0.5 on the holdout (66,358 training Source 1 records,
fixed id hash, never used for learning).
1. Apply exclusivity: each Source 2/3 record is kept only for its highest-probability Source 1
   record.
2. Apply the keep rule: q ≥ t, or top candidate with q ≥ t1. We grid-search t ∈ 0.50–0.90 and
   t1 ∈ 0.40–0.76.
3. Choose (t, t1) with a 2-fold honest check: pick on one half of the records, measure on the
   other half.

The final model uses **FINAL_RULE**. We also compared expected-F0.5 per-record decoding; it was
not better under the honest check.

### Post-submission enhancements (v16 experiment, code shipped switched OFF)

Three changes target the two remaining loss sources (blocking misses and the unsure pairs).
They are present in the development tree but are **disabled in this submission** (`build_submission.py`
reverts them before staging), so `run_pipeline.py` reproduces the submitted files exactly.

1. **Unsearched-best rescue** (`config.py` `BLOCKING.rescue`, `blocking.py`, feature
   `search_margin`): besides the top-k candidates, the best pairs per record that scored
   DIRECTLY below the last kept candidate can be added (no new search — the raw top-k result
   already contains them). The model decides, guided by `search_margin` (the gap down from
   the best kept candidate): a true copy is almost never the worst-looking pair of its own
   record. Attacks the ~7% of records whose true link the searches never propose.
2. **Rival features as a `src/` module** (`rival.py`, 17 columns `rv_*`): the ens3 stage-2
   upgrade (+0.0015 honest on the holdout), previously only embedded in the Kaggle kernel:
   for every unsure pair, the copy is compared with ALL Source 1 records that share its name
   key or its address numbers — the exact spelling (legal form, address numbers) points at
   the owner. This module IS part of the submitted configuration (it replaces the kernel's
   embedded blob and produces the same features).
3. **Stage-1 gap / log-odds features** (removed from the submitted configuration): tried on
   top of the collective set; left out pending the honest 2-fold check.

Each change must clear the same honest holdout + 2-fold check as before; with the rescue on,
the blocking report (pair recall and "best possible macro F0.5", ceiling 0.9925 without)
shows the achievable gain directly.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), holdout of 66,358 training Source 1 records:** **0.9874**
  (stage-2 model: 0.9874, P 0.9982, R 0.9663; India 0.9854, US 0.9887; records with 0 matches
  0.9956, 1 match 0.9578, 2–3 matches 0.9867, 4+ matches 0.9904).
  **Public leaderboard:** **0.983**.

  | version | change | holdout F0.5 | public LB |
  |---|---|---|---|
  | v1 | first LightGBM, 15 candidates | 0.957 | 0.948 |
  | v2 | full data, legal-form features, fixed blocking | 0.9803 | 0.968 |
  | v5 | + XLM-R cross-encoder layer | 0.9825 | 0.975 |
  | v6/v7 | cross-encoder learns from 150k/300k records | 0.9836 | 0.978 |
  | v9 | mean of 3 cross-encoders | 0.9839 | 0.979 |
  | v11 | top-1 keep rule | 0.9842 | 0.980 |
  | v13 | 30 candidates | 0.9849 | 0.9805 |
  | v15 | + rival re-scorer on the decisions | 0.9861 | 0.982 |
  | final | stage 2 with rival features + 5–6 cross-encoders + self-training | 0.9874 | 0.983 |

- **Common false positives (wrong merges):**
  - Near-identical names at the same address that are different businesses ("First Ready Inc"
    vs "Fourth Ready Inc"; "LMK" vs "IMK Interio").
  - Branches of one chain where the Source 2/3 copy has a partial address matching several
    branches.
  - Copies with an added generic word ("… Private Limited Services / Center") that the model
    treats as noise.
- **Common false negatives (missed matches):**
  - Heavily abbreviated names ("Savoia One Inc" → "Sa One Inc"; "Hb Land" → "HB LTD" with no
    address).
  - Copies with a changed house number and a different city spelling (Bangalore/Bengaluru).
  - Names written in another script whose transliteration differs a lot from the English name.
- **Remaining blocking misses:** about 7% of holdout records have a true link that the search
  never proposes. These are mostly native-script names with an address reduced to a city, and
  copies with a missing address and a changed name. Even perfect decisions on our candidates
  would give holdout F0.5 0.9925, so blocking remains the main limit.

---

## 6. Conclusion

A CPU-first pipeline resolves millions of noisy multilingual business records without any
external data. It combines normalisation, bidirectional TF-IDF blocking, a two-stage LightGBM
with collective and rival features, a small ensemble of fine-tuned cross-encoders, self-training
on confident test pairs, and metric-aware decoding. Key lessons:
- Blocking recall caps the score.
- The one-owner structure and rival comparisons fix most wrong merges.
- Improvements must be checked with an honest split. Several ideas that looked good in-sample
  (per-record expected-F0.5 decoding, an LLM judge, re-scoring on top of the final model) did
  not survive the 2-fold check and were left out.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

- `src/run_pipeline.py` is the entry point:
  `python src/run_pipeline.py all --data-dir <dataset> --work-dir work`.
  It runs clean → blocking → features → stage 1 → cross-encoder scores → stage 2 → assign and
  writes `work/output/matching_results.tsv` and `work/output/candidate_pairs.tsv`.
  - S0 read/write: `src/io_utils.py`
  - S1 cleaning: `text_rules.py`, `normalize.py`, `indic.py`
  - S2 blocking: `blocking.py`, `run_blocking.py`
  - S3 features: `features.py`, `learn_name_words.py`
  - stage 1: `train.py`
  - cross-encoder layer: `cross_encoder.py`
  - collective stage 2: `stage2.py`
  - decoding and output: `assign.py`
  - metric and reports: `evaluate.py`
  - all parameters: `config.py`
- `kaggle/kernels/*/run.py` are the exact Kaggle notebook scripts of the final runs:
  - `wide30`: blocking with 30 candidates and stage 1
  - `llm2`–`llm7`: fine-tuning and scoring of the cross-encoders
  - `ens3`: stage 2 with rival features and self-training (`rival.py` is embedded in it)
  - `ens3a` / `ens3c`: assign-only run and second seed
- `kaggle/ctx/`: local scripts for the decision rule.
  - `apply_rule.py`: exclusivity + keep rule (t, t1) applied to a pipeline's pair probabilities.
  - `ctx8.py`: honest 2-fold threshold check and the optional rival re-scorer.
- `README.md` gives the exact commands and run times. `requirements.txt` gives pinned versions.

**Licences of everything used (no external data, no external APIs, no geocoding):**

| package / model | licence | size |
|---|---|---|
| LightGBM | MIT | trees |
| rapidfuzz | MIT | – |
| anyascii | ISC | – |
| sparse_dot_topn | Apache-2.0 | – |
| scikit-learn, numpy, scipy, pandas | BSD-3-Clause | – |
| pyarrow | Apache-2.0 | – |
| PyTorch, transformers | BSD-3 / Apache-2.0 | – |
| FacebookAI/xlm-roberta-base | MIT | 278M params |
| FacebookAI/xlm-roberta-large | MIT | 560M params |
| BAAI/bge-reranker-v2-m3 | Apache-2.0 | 568M params |
| Qwen/Qwen2.5-7B-Instruct-AWQ (tried as an LLM judge, **not** in the final answer) | Apache-2.0 | 7.6B params |

All models have at most 8B parameters and all fine-tuning used only the challenge training data.

### B. Additional Results

- **Blocking recall by country:** India 0.971, US 0.989 (full training set).
- **Holdout by number of true matches** (final stage-2 model): 0 → 0.9956, 1 → 0.9578,
  2–3 → 0.9867, 4+ → 0.9904.
- **Negative experiments** (2-fold honest check on the holdout; none is in the final answer):
  - expected-F0.5 decoding: +0.0018 vs +0.0019 for the threshold rule
  - sibling expansion: +0.00003
  - Qwen2.5-7B one-token judge on unsure pairs: −0.00002
  - re-scoring the final stage 2 with the rival re-scorer: −0.00016
  - blending the final model with the previous one: −0.0001 to −0.0007

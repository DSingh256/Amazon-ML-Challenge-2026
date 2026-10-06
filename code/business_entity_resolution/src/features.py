"""
S3 - Pair features: about 40 numbers that describe how alike a Source 1 record
and one of its candidates are. The model (S4) learns from these numbers.

Groups of features:
  name      : string similarities of the cleaned names (fuzzy, cosine, IDF-weighted overlap)
  legal     : do the legal words (inc, pvt, ltd ...) agree, conflict, or is one missing?
  address   : same kinds of similarities on the address, plus number matching
              (house number, flat number ...)
  retrieval : what the search (S2) already knew: ranks, scores, how many methods found it
  competition: is this candidate the best one of its Source 1 record? Is the Source 1
              record the best one for this candidate?

A feature is NaN (= "unknown") when it cannot be judged, for example every
address feature is NaN when one of the two addresses is empty. LightGBM handles NaN.

Never used as a feature: country, ids (the model must work for unseen countries).

IMPORTANT for training: the "competition" features look at ALL candidates in the table
that is passed in. So training and test must be built the same way: candidates for
*every* Source 1 record of a self-contained world (see make_sample.py), not for a
random slice of Source 1 records.

Usage:
    python src/features.py --work-dir D:/AmazonML/work/sample_run --split train \
                           --data-dir D:/AmazonML/work/sample_data
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

from config import TRAIN_FRACTION
from io_utils import id_bucket, log, read_ground_truth_pairs, truth_as_rows
from text_rules import LEGAL_WORDS

CHUNK_SIZE = 200_000      # pairs handled at once (keeps memory small)
TEXT_CHUNK = 500_000      # records turned into 3-letter pieces at once
N_HASHED_PIECES = 2 ** 22  # 3-letter pieces are numbered by hashing (no vocabulary kept in memory)
FIRST_NUMBER = re.compile(r"\b\d+\b")
# postcode = the last 4-6 digit number that is not the first word (the first word is
# usually the house number). 4 digits too, because leading zeros are removed ("07030").
# Found by pattern only, never by country.
POSTCODE = re.compile(r"(?<=\s)\d{4,6}\b(?!.*\b\d{4,6}\b)")

# "Sound skeleton" of a name: letters that sound alike become one letter, vowels are dropped
# (except the first letter of a word), double letters become one. Spellings of the same
# spoken name then look alike: "shivam ventures" -> "svm vntrs", "shivam venchars"
# (from Hindi script) -> "svm vnCrs", Tamil "kulopal" and "global" -> "klpl".
SOUND_PAIRS = [("chh", "C"), ("ch", "C"), ("sh", "s"), ("ph", "f"), ("kh", "k"), ("gh", "k"),
               ("th", "t"), ("dh", "t"), ("bh", "p"), ("jh", "C"), ("ck", "k"), ("q", "k"),
               ("x", "ks"), ("w", "v"), ("z", "C"), ("j", "C"), ("b", "p"), ("d", "t"), ("g", "k"),
               ("c", "k")]
NOT_FIRST_VOWEL = re.compile(r"(?<=\w)[aeiouyh]")
DOUBLE_LETTER = re.compile(r"(\w)\1+")

# Typos inside name words: a digit in place of a look-alike letter ("5ervices",
# "techno1ogies") and "l" in place of a capital "I" at the start ("lndia", "lnc").
# Only in words with at least 3 letters, so "3rd", "24x7", "b2b" stay as they are.
DIGIT_AS_LETTER = str.maketrans("0134568", "oleasgb")
MIXED_WORD = re.compile(r"\b(?=(?:\d*[a-z]){3})(?=[a-z]*\d)[a-z0-9]+\b")
L_AS_CAPITAL_I = re.compile(r"\bl(?=[nmstdcfvgkpr])")

# Learned from the training ground truth by learn_name_words.py
NAME_WORDS = json.loads((Path(__file__).resolve().parent / "name_words.json").read_text(encoding="utf-8"))
WORD_MAP = NAME_WORDS["word_map"]
NOISE_WORDS = set(NAME_WORDS["noise_words"])

# Legal forms as bits, so a set of legal words is one number ("ltd pvt" = ltd bit + pvt bit).
LEGAL_BITS = {form: 1 << i for i, form in enumerate(sorted(set(LEGAL_WORDS.values())))}
# Long legal words, also found with a typo ("prvate", "limted"): a misspelt legal word
# is noise on a true match, while a really different legal form is a different company.
LONG_LEGAL_WORDS = [w for w in LEGAL_WORDS if len(w) >= 5]


# ---------------------------------------------------------------------------
# Step 1: prepare everything that belongs to single records (done once)
# ---------------------------------------------------------------------------
def row_sums(matrix):
    return np.asarray(matrix.sum(axis=1)).ravel()


def char_tfidf(text1, text23):
    """TF-IDF of 3-letter pieces (same weights as sklearn TfidfVectorizer with
    sublinear_tf: (1 + log count) * idf, rows scaled to length 1), learned from both
    tables. Pieces are numbered by hashing and the texts are handled in chunks, so the
    ~10 million test records never need more memory than the finished matrices.
    Returns (matrix of text1, matrix of text23)."""
    hasher = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_HASHED_PIECES,
                               alternate_sign=False, norm=None, dtype=np.float32)

    def chunks(texts):
        for start in range(0, len(texts), TEXT_CHUNK):
            counts = hasher.transform(texts.iloc[start:start + TEXT_CHUNK])
            counts.sum_duplicates()
            yield start, counts

    # pass 1: in how many records does every piece appear?
    doc_freq = np.zeros(N_HASHED_PIECES, dtype=np.int64)
    sizes = []
    for texts in (text1, text23):
        n_values = 0
        for _, counts in chunks(texts):
            doc_freq += np.bincount(counts.indices, minlength=N_HASHED_PIECES)
            n_values += counts.nnz
        sizes.append(n_values)
    n_records = len(text1) + len(text23)
    idf = (np.log((1 + n_records) / (1 + doc_freq)) + 1).astype(np.float32)

    # pass 2: weights, written straight into the final arrays (no second copy)
    matrices = []
    for texts, n_values in zip((text1, text23), sizes):
        index_type = np.int32 if n_values < 2 ** 31 else np.int64
        data = np.empty(n_values, dtype=np.float32)
        indices = np.empty(n_values, dtype=index_type)
        indptr = np.zeros(len(texts) + 1, dtype=index_type)
        filled = 0
        for start, counts in chunks(texts):
            counts.data = (1 + np.log(counts.data)) * idf[counts.indices]
            counts = normalize(counts)
            data[filled:filled + counts.nnz] = counts.data
            indices[filled:filled + counts.nnz] = counts.indices
            indptr[start + 1:start + 1 + counts.shape[0]] = counts.indptr[1:] + filled
            filled += counts.nnz
        matrices.append(sp.csr_matrix((data, indices, indptr), shape=(len(texts), N_HASHED_PIECES)))
    return matrices


def build_field_matrices(text1, text23):
    """TF-IDF style matrices of one text field (name or address).

    IDF (how rare a word is) is learned from all records of this split.
    Returns a dict of sparse matrices; suffix 1 = Source 1 rows, 23 = Source 2/3 rows.
    """
    both = pd.concat([text1, text23])

    # words: each present word weighs its IDF (rare word = big weight)
    words = TfidfVectorizer(token_pattern=r"\S+", binary=True, norm=None, dtype=np.float32)
    words.fit(both)
    idf1, idf23 = words.transform(text1), words.transform(text23)
    present23 = idf23.copy()
    present23.data[:] = 1

    # 3-letter pieces (finds similarity through typos)
    char1, char23 = char_tfidf(text1, text23)

    return {
        "idf1": idf1, "present23": present23,
        "idf_sum1": row_sums(idf1), "idf_sum23": row_sums(idf23),
        "word1": normalize(idf1), "word23": normalize(idf23),
        "char1": char1, "char23": char23,
    }


def first_number(text):
    match = FIRST_NUMBER.search(text)
    return match.group(0) if match else ""


def postcode(text):
    match = POSTCODE.search(text)
    return match.group(0) if match else ""


def sound_skeleton(name):
    for old, new in SOUND_PAIRS:
        name = name.replace(old, new)
    return DOUBLE_LETTER.sub(r"\1", NOT_FIRST_VOWEL.sub("", name))


def fix_typos(name):
    name = MIXED_WORD.sub(lambda m: m.group(0).translate(DIGIT_AS_LETTER), name)
    return L_AS_CAPITAL_I.sub("i", name)


def name_key(name):
    """Name for the "key" features: typos fixed, Indian-script spellings of English
    words turned into the English word, added noise words ("center", "shri") and long
    numbers (ids, phone numbers) removed. Falls back to the fixed name if nothing is left."""
    words = [WORD_MAP.get(w, w) for w in fix_typos(name).split()]
    key = [w for w in words if w not in NOISE_WORDS and not (w.isdigit() and len(w) >= 5)]
    return " ".join(dict.fromkeys(key or words))


def name_keys(names):
    """name_key of every name (computed once per different name)."""
    codes, uniques = pd.factorize(names.fillna(""))
    return pd.Series(np.array([name_key(n) for n in uniques] + [""], dtype=object)[codes], index=names.index)


def unseen_word_share(names, vocabulary):
    """Share of the words of each name that no Source 1 name uses. Made-up alias names
    ("Miranex", "tiprivate com") are high; then only the address can tell a match."""
    codes, uniques = pd.factorize(names.fillna(""))
    shares = [np.mean([w not in vocabulary for w in n.split()]) if n.split() else 0.0 for n in uniques]
    return np.array(shares + [0.0], dtype=np.float32)[codes]


def legal_bits(forms):
    """legal_form ("ltd pvt") of every record as a bit number (0 = no legal words)."""
    codes, uniques = pd.factorize(forms.fillna(""))
    bits = [sum(LEGAL_BITS[f] for f in u.split()) for u in uniques]
    return np.array(bits + [0], dtype=np.int64)[codes]


def word_bits(names, word_to_bits):
    """OR of word_to_bits(word) over the words of every name (each different word and
    name is looked at once)."""
    cache = {}
    def bits(name):
        total = 0
        for w in name.split():
            if w not in cache:
                cache[w] = word_to_bits(w)
            total |= cache[w]
        return total
    codes, uniques = pd.factorize(names.fillna(""))
    return np.array([bits(u) for u in uniques] + [0], dtype=np.int64)[codes]


def legal_word_bits(word):
    """Bit of a legal word, also when written with a typo (one or two letters off)."""
    if word in LEGAL_WORDS:
        return LEGAL_BITS[LEGAL_WORDS[word]]
    if len(word) >= 5:
        best = process.extractOne(word, LONG_LEGAL_WORDS, scorer=fuzz.ratio, score_cutoff=85)
        if best is not None:
            return LEGAL_BITS[LEGAL_WORDS[best[0]]]
    return 0


def bit_count(values):
    values = np.asarray(values, dtype=np.int64)
    return sum((values >> k) & 1 for k in range(63)).astype(np.float32)


def initials(key):
    """First letters of the words of a name key ("holistic sons" -> "hs"); "" for one word."""
    words = key.split()
    return "".join(w[0] for w in words) if len(words) >= 2 else ""


def sound_skeletons(names):
    """sound_skeleton of every name (computed once per different name)."""
    codes, uniques = pd.factorize(names.fillna(""))
    return np.array([sound_skeleton(n) for n in uniques] + [""], dtype=object)[codes]


class FeatureContext:
    """Everything about single records that the pair features need."""

    def __init__(self, s1, s23, s1_vocabulary=frozenset()):
        self.s1, self.s23 = s1, s23
        self.unseen23 = unseen_word_share(s23["name_core"], s1_vocabulary)
        self.name = build_field_matrices(s1["name_core"], s23["name_core"])
        self.addr = build_field_matrices(s1["addr_clean"], s23["addr_clean"])

        # numbers inside the address (house no., flat no., ...)
        numbers = CountVectorizer(token_pattern=r"\b\d+\b", binary=True, dtype=np.float32)
        numbers.fit(pd.concat([s1["addr_clean"], s23["addr_clean"]]))
        self.numbers1 = numbers.transform(s1["addr_clean"])
        self.numbers23 = numbers.transform(s23["addr_clean"])
        self.first_number1 = s1["addr_clean"].map(first_number).to_numpy()
        self.first_number23 = s23["addr_clean"].map(first_number).to_numpy()
        self.postcode1 = s1["addr_clean"].map(postcode).to_numpy()
        self.postcode23 = s23["addr_clean"].map(postcode).to_numpy()

        # first / last word of the name
        self.first_word1 = s1["name_core"].str.split().str[0].fillna("").to_numpy()
        self.first_word23 = s23["name_core"].str.split().str[0].fillna("").to_numpy()
        self.last_word1 = s1["name_core"].str.split().str[-1].fillna("").to_numpy()
        self.last_word23 = s23["name_core"].str.split().str[-1].fillna("").to_numpy()

        # sound skeletons of the names (spelling / script differences, see sound_skeleton)
        self.skeleton1 = sound_skeletons(s1["name_core"])
        self.skeleton23 = sound_skeletons(s23["name_core"])

        # name keys (see name_key) and their IDF-weighted words
        self.key1, self.key23 = name_keys(s1["name_core"]), name_keys(s23["name_core"])
        words = TfidfVectorizer(token_pattern=r"\S+", binary=True, norm=None, dtype=np.float32)
        words.fit(pd.concat([self.key1, self.key23]))
        key_idf1, key_idf23 = words.transform(self.key1), words.transform(self.key23)
        self.key_idf_sum1, self.key_idf_sum23 = row_sums(key_idf1), row_sums(key_idf23)
        self.key_present23 = key_idf23.copy()
        self.key_present23.data[:] = 1
        self.key_idf1, self.key_word1, self.key_word23 = key_idf1, normalize(key_idf1), normalize(key_idf23)
        self.key1, self.key23 = self.key1.to_numpy(), self.key23.to_numpy()

        # legal words as bit numbers (exact, and with typos allowed)
        self.legal1, self.legal23 = legal_bits(s1["legal_form"]), legal_bits(s23["legal_form"])
        self.fuzzy_legal1 = word_bits(s1["name_clean"], legal_word_bits)
        self.fuzzy_legal23 = word_bits(s23["name_clean"], legal_word_bits)
        # noise words ("center", "service" ...) of a name, as bits
        noise_bit = {w: 1 << i for i, w in enumerate(sorted(NOISE_WORDS)[:63])}
        self.noise1 = word_bits(s1["name_core"], lambda w: noise_bit.get(w, 0))
        self.noise23 = word_bits(s23["name_core"], lambda w: noise_bit.get(w, 0))


# ---------------------------------------------------------------------------
# Step 2: features of pairs (done chunk by chunk)
# ---------------------------------------------------------------------------
def rowwise_dot(matrix1, matrix2, a, b):
    """For every pair i: dot product of row a[i] of matrix1 and row b[i] of matrix2."""
    return np.asarray(matrix1[a].multiply(matrix2[b]).sum(axis=1)).ravel()


def safe_divide(top, bottom):
    """top / bottom, NaN where bottom is 0."""
    return np.divide(top, bottom, out=np.full(len(top), np.nan, dtype=np.float32), where=bottom > 0)


def string_scores(texts1, texts23, scorer):
    """Fuzzy score of every pair of texts (all CPU cores)."""
    return process.cpdist(texts1, texts23, scorer=scorer, dtype=np.float32, workers=-1)


def field_overlap_features(matrices, prefix, a, b):
    """Cosine and IDF-weighted overlap of one text field."""
    shared = rowwise_dot(matrices["idf1"], matrices["present23"], a, b)
    total = matrices["idf_sum1"][a] + matrices["idf_sum23"][b] - shared
    return {
        f"{prefix}_word_cosine": rowwise_dot(matrices["word1"], matrices["word23"], a, b),
        f"{prefix}_char_cosine": rowwise_dot(matrices["char1"], matrices["char23"], a, b),
        f"{prefix}_idf_jaccard": safe_divide(shared, total),
    }


def name_features(ctx, a, b):
    core1 = ctx.s1["name_core"].to_numpy()[a].tolist()
    core23 = ctx.s23["name_core"].to_numpy()[b].tolist()
    clean1 = ctx.s1["name_clean"].to_numpy()[a].tolist()
    clean23 = ctx.s23["name_clean"].to_numpy()[b].tolist()

    features = {
        "name_ratio": string_scores(core1, core23, fuzz.ratio),
        "name_token_set": string_scores(core1, core23, fuzz.token_set_ratio),
        "name_token_sort": string_scores(core1, core23, fuzz.token_sort_ratio),
        "name_partial": string_scores(core1, core23, fuzz.partial_ratio),
        "name_jaro_winkler": string_scores(core1, core23, JaroWinkler.normalized_similarity),
        "name_levenshtein": string_scores(core1, core23, Levenshtein.normalized_similarity),
        "name_full_ratio": string_scores(clean1, clean23, fuzz.ratio),   # legal words included
    }
    features.update(field_overlap_features(ctx.name, "name", a, b))
    skeleton1, skeleton23 = ctx.skeleton1[a].tolist(), ctx.skeleton23[b].tolist()
    features["name_sound_ratio"] = string_scores(skeleton1, skeleton23, fuzz.ratio)
    features["name_sound_token_set"] = string_scores(skeleton1, skeleton23, fuzz.token_set_ratio)

    len1 = ctx.s1["name_core"].str.len().to_numpy()[a]
    len23 = ctx.s23["name_core"].str.len().to_numpy()[b]
    words1 = ctx.s1["name_core"].str.count(" ").to_numpy()[a]
    words23 = ctx.s23["name_core"].str.count(" ").to_numpy()[b]
    features["name_length_diff"] = np.abs(len1 - len23).astype(np.float32)
    features["name_word_count_diff"] = np.abs(words1 - words23).astype(np.float32)
    features["name_first_word_equal"] = (ctx.first_word1[a] == ctx.first_word23[b]).astype(np.float32)
    features["name_last_word_equal"] = (ctx.last_word1[a] == ctx.last_word23[b]).astype(np.float32)
    features["name_exact_equal"] = (np.array(core1) == np.array(core23)).astype(np.float32)
    features.update(key_features(ctx, a, b))
    return features


def key_features(ctx, a, b):
    """Name keys: noise words removed, Indian-script spellings mapped to English.
    Besides overlap, the IDF mass of words found on only ONE side: a rare word that
    the other name lacks is strong evidence against a match, a common one is not."""
    key1, key23 = ctx.key1[a].tolist(), ctx.key23[b].tolist()
    shared = rowwise_dot(ctx.key_idf1, ctx.key_present23, a, b)
    sum1, sum23 = ctx.key_idf_sum1[a], ctx.key_idf_sum23[b]
    # spaces removed: "templepresbyterianchurch" vs "temple presbyterian church"
    joined1, joined23 = [k.replace(" ", "") for k in key1], [k.replace(" ", "") for k in key23]
    # a short form made of the first letters ("hsons com" for "holistic sons")
    acronym = [bool(i) and k.startswith(i) for i, k in zip(map(initials, key1), key23)]
    return {
        "key_joined_ratio": string_scores(joined1, joined23, fuzz.ratio),
        "key_joined_partial": string_scores(joined1, joined23, fuzz.partial_ratio),
        "key_acronym": np.array(acronym, dtype=np.float32),
        "s23_unseen_words": ctx.unseen23[b],
        "key_ratio": string_scores(key1, key23, fuzz.ratio),
        "key_token_set": string_scores(key1, key23, fuzz.token_set_ratio),
        "key_word_cosine": rowwise_dot(ctx.key_word1, ctx.key_word23, a, b),
        "key_idf_jaccard": safe_divide(shared, sum1 + sum23 - shared),
        "key_containment": safe_divide(shared, np.minimum(sum1, sum23)),
        "key_only_s1_idf": (sum1 - shared).astype(np.float32),
        "key_only_s23_idf": (sum23 - shared).astype(np.float32),
        "key_equal": (np.array(key1) == np.array(key23)).astype(np.float32),
    }


def legal_features(ctx, a, b):
    legal1 = ctx.s1["legal_form"].to_numpy()[a]
    legal23 = ctx.s23["legal_form"].to_numpy()[b]
    both_known = (legal1 != "") & (legal23 != "")
    # Which legal forms, and what changed. A different company with the same name and
    # address often has a different legal form ("Private Limited" -> "Limited", "LLC" -> "Ltd");
    # the plain equal / conflict flags cannot tell which change it is.
    bits1, bits23 = ctx.legal1[a], ctx.legal23[b]
    fuzzy1, fuzzy23 = ctx.fuzzy_legal1[a], ctx.fuzzy_legal23[b]
    fuzzy_known = (fuzzy1 > 0) & (fuzzy23 > 0)
    return {
        "legal_equal": (both_known & (legal1 == legal23)).astype(np.float32),
        "legal_conflict": (both_known & (legal1 != legal23)).astype(np.float32),
        "legal_any_missing": (~both_known).astype(np.float32),
        "legal_s1_only": ((bits1 > 0) & (bits23 == 0)).astype(np.float32),
        "legal_s23_only": ((bits1 == 0) & (bits23 > 0)).astype(np.float32),
        "legal_form_s1": bits1.astype(np.float32),
        "legal_form_s23": bits23.astype(np.float32),
        "legal_fuzzy_equal": (fuzzy_known & (fuzzy1 == fuzzy23)).astype(np.float32),
        "legal_dropped": np.where(fuzzy_known, bit_count(fuzzy1 & ~fuzzy23), np.nan),
        "legal_added": np.where(fuzzy_known, bit_count(fuzzy23 & ~fuzzy1), np.nan),
        "name_noise_added": bit_count(ctx.noise23[b] & ~ctx.noise1[a]),
    }


def address_features(ctx, a, b):
    text1 = ctx.s1["addr_clean"].to_numpy()[a]
    text23 = ctx.s23["addr_clean"].to_numpy()[b]
    missing1, missing23 = text1 == "", text23 == ""
    unknown = missing1 | missing23            # cannot compare: features become NaN

    features = {
        "addr_missing_s1": missing1.astype(np.float32),
        "addr_missing_s23": missing23.astype(np.float32),
    }
    fuzzy = {
        "addr_ratio": fuzz.ratio, "addr_token_set": fuzz.token_set_ratio,
        "addr_token_sort": fuzz.token_sort_ratio, "addr_partial": fuzz.partial_ratio,
    }
    for name, scorer in fuzzy.items():
        features[name] = string_scores(text1.tolist(), text23.tolist(), scorer)
    features.update(field_overlap_features(ctx.addr, "addr", a, b))

    # numbers: how many numbers do the two addresses share?
    shared = rowwise_dot(ctx.numbers1, ctx.numbers23, a, b)
    count1 = row_sums(ctx.numbers1[a])
    count23 = row_sums(ctx.numbers23[b])
    features["num_shared"] = shared
    features["num_jaccard"] = safe_divide(shared, count1 + count23 - shared)
    features["num_conflict"] = ((count1 > 0) & (count23 > 0) & (shared == 0)).astype(np.float32)
    first1, first23 = ctx.first_number1[a], ctx.first_number23[b]
    known = (first1 != "") & (first23 != "")
    features["first_number_equal"] = np.where(known, (first1 == first23).astype(np.float32), np.nan)
    # postcode: 1 equal, 0 conflict, NaN = missing on one side
    code1, code23 = ctx.postcode1[a], ctx.postcode23[b]
    known = (code1 != "") & (code23 != "")
    features["postcode_equal"] = np.where(known, (code1 == code23).astype(np.float32), np.nan)

    for name in list(features):
        if name.startswith("addr_") and name not in ("addr_missing_s1", "addr_missing_s23"):
            features[name] = np.where(unknown, np.nan, features[name])
    for name in ("num_shared", "num_jaccard", "num_conflict", "first_number_equal", "postcode_equal"):
        features[name] = np.where(unknown, np.nan, features[name])
    return features


def pair_features(ctx, a, b):
    """All record-level features for the pairs (a[i], b[i]) as a dict of arrays."""
    features = {}
    features.update(name_features(ctx, a, b))
    features.update(legal_features(ctx, a, b))
    features.update(address_features(ctx, a, b))
    return features


# ---------------------------------------------------------------------------
# Step 3: features that compare candidates with each other (need the whole table)
# ---------------------------------------------------------------------------
def rank_in_group(groups, score):
    """1 for the highest score inside each group, 2 for the next ... (ties: first row first).
    groups: list of integer arrays that together define the group."""
    order = np.lexsort((np.arange(len(score)), -score, *reversed(groups)))
    new_group = np.zeros(len(order), dtype=bool)
    new_group[:1] = True
    for g in groups:
        sorted_key = g[order]
        new_group[1:] |= sorted_key[1:] != sorted_key[:-1]
    group_start = np.maximum.accumulate(np.where(new_group, np.arange(len(order)), 0))
    rank = np.empty(len(order), dtype=np.float32)
    rank[order] = np.arange(len(order)) - group_start + 1
    return rank


def retrieval_and_competition_features(candidates, s23_source, s23_name, positions):
    """Features from the candidate table itself (S2 output).
    They look at ALL candidates, but are returned only for the rows `positions`.
    Written with plain arrays: the full test table has ~50 million candidates."""
    c = candidates
    s1 = c["s1_row"].to_numpy()
    s23 = c["s23_row"].to_numpy()
    score = c["best_score"].to_numpy()
    source = s23_source[s23]
    out = {}
    out["n_candidates_for_s1"] = np.bincount(s1)[s1][positions]
    # near-duplicates: other candidates of the same Source 1 record with the same cleaned
    # name (Source 2 and Source 3 copies of one business often look alike)
    name_code = pd.factorize(s23_name)[0][s23]
    _, same_name, count = np.unique(s1.astype(np.int64) * (int(name_code.max()) + 1) + name_code,
                                    return_inverse=True, return_counts=True)
    out["n_same_name_candidates"] = count[same_name[positions]] - 1
    del name_code, same_name, count
    for column in ["candidate_rank", "best_rank", "best_score", "n_methods"]:
        out[column] = c[column].to_numpy()[positions]
    # search_margin: NaN = the search kept this pair; a number = it was NOT in the top-k lists
    # but only `rescue`d (blocking.py): the gap down from the record's best kept candidate.
    # Lets the model separate rescue pairs that are almost as good as the kept ones (where real
    # matches hide) from the ones far below (almost never matches).
    out["search_margin"] = c["search_margin"].to_numpy()[positions] if "search_margin" in c.columns else np.full(len(positions), np.nan, dtype=np.float32)
    for column in c.columns:
        if column.startswith("score_"):
            out[column] = c[column].to_numpy()[positions]

    out["source"] = source[positions]

    # how far behind the best candidate of the same Source 1 record?
    best_for_s1 = pd.Series(score).groupby(s1).transform("max").to_numpy()
    out["gap_to_best_for_s1"] = (best_for_s1 - score)[positions]
    del best_for_s1
    # rank among candidates of the same Source 1 record AND same source (2 or 3)
    out["source_rank"] = rank_in_group([s1, source], score)[positions]

    # the other direction: how good is this Source 1 record for the Source 2/3 record?
    out["reverse_rank"] = rank_in_group([s23], score)[positions]
    best_for_s23 = pd.Series(score).groupby(s23).transform("max").to_numpy()
    out["gap_to_best_for_s23"] = (best_for_s23 - score)[positions]
    del best_for_s23
    out["n_s1_wanting_this"] = np.bincount(s23)[s23][positions]
    return pd.DataFrame({name: values.astype(np.float32) for name, values in out.items()})


# ---------------------------------------------------------------------------
# Step 4: everything together, written to disk chunk by chunk
# ---------------------------------------------------------------------------
def only_used_rows(table, rows):
    """Keep only the records that appear in `rows` (saves memory).
    Returns the smaller table and the new row number of every entry of `rows`."""
    used = np.unique(rows)
    return table.iloc[used].reset_index(drop=True), np.searchsorted(used, rows)


def pair_key(s1_rows, s23_rows):
    """One int64 number per pair (fast membership tests)."""
    return s1_rows.astype(np.int64) * 100_000_000 + s23_rows.astype(np.int64)


def write_features(s1, s23, candidates, out_path, keep_s1=None, truth=None):
    """Compute features for candidate pairs and write them to out_path (parquet).

    keep_s1 : optional True/False per Source 1 row; only pairs of those records are
              written (competition features still use ALL candidates)
    truth   : optional true pairs (s1_row, s23_row); adds the column 'label'
    Returns the number of pairs written.
    """
    assert (s1.index.to_numpy() == np.arange(len(s1))).all(), "s1 needs a 0..n-1 index"
    assert (s23.index.to_numpy() == np.arange(len(s23))).all(), "s23 needs a 0..n-1 index"
    positions = np.arange(len(candidates))
    if keep_s1 is not None:
        positions = positions[keep_s1[candidates["s1_row"].to_numpy()]]
    a_all = candidates["s1_row"].to_numpy()[positions]
    b_all = candidates["s23_row"].to_numpy()[positions]

    # competition features need every candidate
    context = retrieval_and_competition_features(candidates, s23["source"].to_numpy(), s23["name_core"],
                                                 positions)
    log("retrieval / competition features ready")
    true_keys = None if truth is None else pair_key(truth["s1_row"].to_numpy(), truth["s23_row"].to_numpy())

    s1_used, a_local = only_used_rows(s1, a_all)
    s23_used, b_local = only_used_rows(s23, b_all)
    s1_vocabulary = frozenset(word for name in s1["name_core"].fillna("").unique() for word in name.split())
    ctx = FeatureContext(s1_used, s23_used, s1_vocabulary)
    del s1_vocabulary
    log(f"record-level matrices ready ({len(s1_used):,} + {len(s23_used):,} records)")

    writer = None
    for start in range(0, len(positions), CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, len(positions))
        chunk = pd.DataFrame({"s1_row": a_all[start:stop].astype(np.int32),
                              "s23_row": b_all[start:stop].astype(np.int32)})
        record = pd.DataFrame(pair_features(ctx, a_local[start:stop], b_local[start:stop]))
        chunk = pd.concat([chunk, record.astype(np.float32),
                           context.iloc[start:stop].reset_index(drop=True)], axis=1)
        if true_keys is not None:
            chunk["label"] = np.isin(pair_key(chunk["s1_row"].to_numpy(), chunk["s23_row"].to_numpy()),
                                     true_keys).astype(np.int8)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)
        log(f"  features for pairs {stop:,} / {len(positions):,}")
    if writer is not None:
        writer.close()
    return len(positions)


def feature_columns(columns):
    """The columns the model may use (everything except keys and label)."""
    not_features = {"s1_row", "s23_row", "label"}
    # The legal-form bit numbers are left out: they only mean something for forms seen in
    # training, and on a country not in the training data they made the model worse
    # (unseen-country check: US learned from India only 0.9504 with them, 0.9726 without).
    not_features |= {"legal_form_s1", "legal_form_s23"}
    return [c for c in columns if c not in not_features]


def training_records(s1, fraction):
    """True for the fixed random part of Source 1 used for training."""
    if fraction >= 1:
        return np.ones(len(s1), dtype=bool)
    return id_bucket(s1["entity_id"]) < fraction * 1000


def make_features(work_dir, split, data_dir=None, train_fraction=TRAIN_FRACTION):
    """Read the cleaned tables + candidates of one split and write features.parquet."""
    work_dir = Path(work_dir)
    # only the columns the features use (the full test tables are big)
    text_columns = ["name_core", "name_clean", "addr_clean", "legal_form"]
    ids = ["entity_id"] if split == "train" else []
    s1 = pd.read_parquet(work_dir / "s1_clean.parquet", columns=text_columns + ids)
    s23 = pd.read_parquet(work_dir / "s23_clean.parquet", columns=text_columns + ["source"] + ids)
    candidates = pd.read_parquet(work_dir / "candidates.parquet")
    log(f"{len(candidates):,} candidate pairs")

    keep_s1, truth = None, None
    if split == "train":
        keep_s1 = training_records(s1, train_fraction)
        truth = truth_as_rows(read_ground_truth_pairs(data_dir), s1, s23)
        log(f"training on {keep_s1.sum():,} of {len(s1):,} Source 1 records")

    n = write_features(s1, s23, candidates, work_dir / "features.parquet", keep_s1, truth)
    (work_dir / "features_info.json").write_text(json.dumps({"train_fraction": train_fraction, "pairs": n}))
    log(f"saved features of {n:,} pairs to {work_dir / 'features.parquet'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--data-dir", help="needed for train: contains train/train_ground_truth.tsv")
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION,
                        help="train only: part of Source 1 records to compute features for")
    args = parser.parse_args()
    make_features(Path(args.work_dir) / args.split, args.split, args.data_dir, args.train_fraction)


if __name__ == "__main__":
    main()

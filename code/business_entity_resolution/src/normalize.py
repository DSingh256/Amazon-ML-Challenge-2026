"""
S1 - Clean business names and addresses so that equal things look equal.

Examples:
    "-- Holloway Peak Inc Seafood"        -> name_core "holloway peak seafood", legal "inc"
    "राम मार्केटिंग प्राइवेट लिमिटेड"          -> name_core "ram marketing",          legal "ltd pvt"
    "01018 KENWOOD ST, HAMMOND, IN"       -> "1018 kenwood street hammond in"
    "1018 Kenwood St, Hammond, Indiana"   -> "1018 kenwood street hammond in"

Main function: normalize_table(df) adds the cleaned columns to a table.
"""
import re
import unicodedata
from multiprocessing import Pool

import pandas as pd
from anyascii import anyascii

from indic import indic_to_latin
from text_rules import (
    ADDRESS_FILLER_WORDS,
    ADDRESS_WORDS,
    LEGAL_WORDS,
    LETTER_SEQUENCES,
    STATE_NAMES,
)

# ---------------------------------------------------------------------------
# Regular expressions (built once, used millions of times)
# ---------------------------------------------------------------------------
# "www.shivshakti.com" or "bethchapel.com" -> keep only "shivshakti" / "bethchapel"
WEBSITE = re.compile(r"(https?://)?(www\.)?([a-z0-9-]+)\.(com|net|org|in|co|biz|info|fr|us|io)(\.[a-z]{2})?\b")

# A dot between two single letters, like in "l.l.c" -> "llc"
DOT_IN_INITIALS = re.compile(r"(?<=\b[a-z])\.(?=[a-z]\b)")

# Anything that is not a letter or a digit
NOT_LETTER_OR_DIGIT = re.compile(r"[^a-z0-9]+")

# Leading zeros of a number: "01018" -> "1018", "005" -> "5"
LEADING_ZEROS = re.compile(r"\b0+(?=\d)")

# A zero used as the letter o inside a word: "pr0perties" -> "properties"
ZERO_AS_LETTER_O = re.compile(r"(?<=[a-z])0(?=[a-z])")

# "M/s", "Messrs" in front of a name ("M/S Shree Ganesh Traders"): not part of the name
NAME_PREFIX = re.compile(r"^(m s|messrs)\s+(?=\S)")


def phrase_pattern(phrases):
    """One regex that finds any of the given phrases as whole words (longest first)."""
    ordered = sorted(phrases, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(p) for p in ordered) + r")\b")


LETTER_SEQUENCE_PATTERN = phrase_pattern(LETTER_SEQUENCES)
STATE_PATTERN = phrase_pattern(STATE_NAMES)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def basic_clean(text: str) -> str:
    """Steps shared by names and addresses.

    1. unify unicode look-alikes          (NFKC)
    2. any script -> latin letters        ("प्राइवेट" -> "praivet", "é" -> "e");
       Indian scripts with their unwritten vowels (indic.py: "शिवम" -> "shivam")
    3. lowercase, "&" -> "and"
    4. remove websites to their main word
    5. punctuation -> space (but "l.l.c" -> "llc" and "lord's" -> "lords")
    6. remove leading zeros of numbers
    """
    text = unicodedata.normalize("NFKC", text)
    text = anyascii(indic_to_latin(text)).lower()
    text = text.replace("&", " and ")
    text = WEBSITE.sub(r" \3 ", text)
    text = DOT_IN_INITIALS.sub("", text)
    text = text.replace("'", "")
    text = NOT_LETTER_OR_DIGIT.sub(" ", text)
    text = LEADING_ZEROS.sub("", text)
    return " ".join(text.split())


def unique_in_order(words):
    """Drop repeated words but keep the order: "a b a c" -> "a b c"."""
    return list(dict.fromkeys(words))


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------
def clean_name(raw: str):
    """Return (name_clean, name_core, legal_form).

    name_clean : whole cleaned name
    name_core  : name without legal words (inc, pvt, ltd, sarl ...)
    legal_form : the legal words in standard short form, sorted ("ltd pvt")
    """
    text = basic_clean(raw)
    text = NAME_PREFIX.sub("", text)
    text = ZERO_AS_LETTER_O.sub("o", text)
    text = LETTER_SEQUENCE_PATTERN.sub(lambda m: LETTER_SEQUENCES[m.group(0)], text)
    words = unique_in_order(text.split())

    legal = sorted({LEGAL_WORDS[w] for w in words if w in LEGAL_WORDS})
    core = [w for w in words if w not in LEGAL_WORDS]
    if not core:              # the name is ONLY legal words: keep it as it is
        core = words

    return " ".join(words), " ".join(core), " ".join(legal)


# ---------------------------------------------------------------------------
# Addresses
# ---------------------------------------------------------------------------
def clean_address(raw: str) -> str:
    """Cleaned address: state names -> codes, short forms -> long forms,
    filler words removed, repeated words removed."""
    text = basic_clean(raw)
    text = STATE_PATTERN.sub(lambda m: STATE_NAMES[m.group(0)], text)   # states FIRST
    words = [ADDRESS_WORDS.get(w, w) for w in text.split()]             # then short forms
    words = [w for w in words if w not in ADDRESS_FILLER_WORDS]
    return " ".join(unique_in_order(words))


# ---------------------------------------------------------------------------
# Whole table
# ---------------------------------------------------------------------------
def apply_to_unique_values(func, values, n_jobs):
    """Run func once per DIFFERENT value (many values repeat), then spread
    the results back to every row. Uses several CPU cores when n_jobs > 1."""
    codes, uniques = pd.factorize(pd.Series(values), sort=False)
    uniques = list(uniques)
    if n_jobs > 1 and len(uniques) > 50_000:
        with Pool(n_jobs) as pool:
            results = pool.map(func, uniques, chunksize=5_000)
    else:
        results = [func(value) for value in uniques]
    return [results[code] for code in codes]


def normalize_table(df: pd.DataFrame, n_jobs: int = 1) -> pd.DataFrame:
    """Add cleaned columns to a table of records.

    New columns:
        name_clean, name_core, legal_form, addr_clean  (see functions above)
        name_text, addr_text, name_addr_text           (texts used for searching)
    """
    df = df.copy()

    names = apply_to_unique_values(clean_name, df["business_name"], n_jobs)
    df["name_clean"] = [n[0] for n in names]
    df["name_core"] = [n[1] for n in names]
    df["legal_form"] = [n[2] for n in names]

    df["addr_clean"] = apply_to_unique_values(clean_address, df["business_address"], n_jobs)

    df["name_text"] = df["name_core"]
    df["addr_text"] = df["addr_clean"]
    df["name_addr_text"] = (df["name_core"] + " " + df["addr_clean"]).str.strip()
    return df

"""
Learn two small word lists from the TRAINING ground truth (no outside data):

  word_map    : a name word written from an Indian script -> the English word it
                stands for in the matching record ("intarneshnal" -> "international",
                "injiniyring" -> "engineering"). Learned from true pairs where one
                name is in an Indian script and the other in Latin letters.
  noise_words : words that are usually ADDED to one side of a true pair and are
                not part of the business name ("center", "services", "shri", "mr").

features.py uses both to build a "name key" (see name_key) that is compared
in extra name features. Nothing here depends on the country of a record.

Usage (writes src/name_words.json):
    python src/learn_name_words.py --data-dir D:/AmazonML/work/dense_data \
                                   --clean-dir D:/AmazonML/work/dense_run3/train
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from config import HOLDOUT_PERCENT
from features import fix_typos, sound_skeleton
from io_utils import id_bucket, log, read_ground_truth_pairs

OUT_FILE = Path(__file__).resolve().parent / "name_words.json"
INDIAN_SCRIPT = re.compile(r"[\u0900-\u0DFF]")
MIN_COUNT = 20            # a word must be seen this often to enter a list
MAP_SHARE = 0.5           # the English partner must explain at least half of the cases
MIN_SOUND = 60            # the two words must sound alike (fuzzy ratio of sound skeletons)
NOISE_SHARE = 0.6         # a noise word is missing on the other side in >= 60% of its uses


def learn(pairs):
    """pairs: DataFrame with name_core_1, name_core_23, indic_1, indic_23 (one row per true pair)."""
    together, extra_count = Counter(), Counter()
    for core1, core23, indic1, indic23 in pairs[["name_core_1", "name_core_23", "indic_1", "indic_23"]].itertuples(index=False):
        if indic1 == indic23:
            continue
        indian, latin = (core1, core23) if indic1 else (core23, core1)
        indian_words, latin_words = set(indian.split()), set(fix_typos(latin).split())
        extra, missing = indian_words - latin_words, latin_words - indian_words
        if not extra or not missing or len(extra) > 4 or len(missing) > 4:
            continue
        for word in extra:
            extra_count[word] += 1
            for partner in missing:
                together[word, partner] += 1

    best = {}
    for (word, partner), count in together.items():
        if count >= MIN_COUNT and count >= MAP_SHARE * extra_count[word] and \
                fuzz.ratio(sound_skeleton(word), sound_skeleton(partner)) >= MIN_SOUND:
            if count > best.get(word, ("", 0))[1]:
                best[word] = (partner, count)
    word_map = {word: partner for word, (partner, _) in best.items() if word != partner}

    # noise words: learned only from pairs where both names are in Latin letters,
    # so that unmapped Indian-script spellings are not taken for noise
    used, missing_other = Counter(), Counter()
    latin = pairs[~pairs.indic_1 & ~pairs.indic_23]
    for core1, core23 in latin[["name_core_1", "name_core_23"]].itertuples(index=False):
        words1, words23 = ([word_map.get(w, w) for w in fix_typos(core).split()] for core in (core1, core23))
        set1, set23 = set(words1), set(words23)
        for words, other in ((set1, set23), (set23, set1)):
            for w in words:
                used[w] += 1
                if w not in other:
                    missing_other[w] += 1
    noise = sorted(w for w, n in used.items()
                   if n >= MIN_COUNT and missing_other[w] >= NOISE_SHARE * n and not w.isdigit())
    return word_map, noise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, nargs="+", help="one or more train samples")
    parser.add_argument("--clean-dir", required=True, nargs="+",
                        help="for each data dir: the folder with its s1_clean / s23_clean")
    args = parser.parse_args()

    parts = []
    for data_dir, clean_dir in zip(args.data_dir, args.clean_dir):
        clean = Path(clean_dir)
        s1 = pd.read_parquet(clean / "s1_clean.parquet", columns=["entity_id", "business_name", "name_core"])
        s23 = pd.read_parquet(clean / "s23_clean.parquet", columns=["entity_id", "business_name", "name_core"])
        truth = read_ground_truth_pairs(data_dir)
        # never learn from the holdout records (train.py measures on them)
        truth = truth[id_bucket(truth["s1_id"]) % 100 >= HOLDOUT_PERCENT]
        parts.append(truth.merge(s1.add_suffix("_1"), left_on="s1_id", right_on="entity_id_1")
                          .merge(s23.add_suffix("_23"), left_on="s23_id", right_on="entity_id_23"))
    pairs = pd.concat(parts).drop_duplicates(["s1_id", "s23_id"])
    for side in ("1", "23"):
        pairs[f"indic_{side}"] = pairs[f"business_name_{side}"].str.contains(INDIAN_SCRIPT)
    log(f"{len(pairs):,} true pairs, {int((pairs.indic_1 != pairs.indic_23).sum()):,} with one name in an Indian script")

    word_map, noise = learn(pairs)
    OUT_FILE.write_text(json.dumps({"word_map": dict(sorted(word_map.items())), "noise_words": noise},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"{len(word_map)} mapped words, {len(noise)} noise words -> {OUT_FILE}")
    log("noise words: " + " ".join(noise))


if __name__ == "__main__":
    main()

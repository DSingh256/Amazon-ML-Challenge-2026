"""
Indian scripts -> latin letters, written the way Indian names are usually spelled in English.

About 20% of the Indian Source 2/3 names are written in an Indian script (Hindi, Telugu,
Kannada, Tamil, Gujarati, Bengali, Malayalam, Oriya, Punjabi). anyascii writes them
letter by letter WITHOUT the vowel "a" that Indian scripts do not write:
    "शिवम वेंचर्स"  -> anyascii "sivm vemcrs"     here: "shivam venchars"
and "sivm vemcrs" shares almost no 3-letter pieces with "shivam ventures", so the search
did not find these records and the model gave them ~0.

All nine scripts use the same Unicode layout (letter k is at the same place in every
block), so one table serves all of them. Rules:
  - a consonant carries the vowel "a" unless a vowel sign or the "no vowel" sign follows
  - that "a" is dropped at the end of a word and between two syllables (Hindi schwa rule,
    Ohala 1983: V C a C V -> V C C V, applied from right to left): "मार्केटिंग" -> "marketing";
    it stays at the end after two consonants of which the last is y r v n m l:
    "कृष्ण" -> "krishna", "सूर्य" -> "surya", "धर्म" -> "dharma" (but "ट्रेडर्स" -> "tredars")
  - the South Indian scripts (Tamil, Telugu, Kannada, Malayalam) say every "a" that is not
    silenced by the "no vowel" sign: "ಕೃಷ್ಣ" -> "krishna", but "ಮಹೇಶ್" -> "mahesh"
  - the nasal dot (anusvara) is "n", or "m" before p / b / m: "कंपनी" -> "kampani"
Everything that is not an Indian script is left as it is (anyascii handles it later).
"""

FIRST, LAST = 0x0900, 0x0D7F

# position inside a script block -> latin
VOWELS = {0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u", 0x0B: "ri",
          0x0C: "li", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o", 0x12: "o", 0x13: "o",
          0x14: "au", 0x60: "ri", 0x61: "li", 0x72: "", 0x73: "u"}
CONSONANTS = {0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "n", 0x1A: "ch", 0x1B: "chh",
              0x1C: "j", 0x1D: "jh", 0x1E: "n", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
              0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
              0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
              0x31: "r", 0x32: "l", 0x33: "l", 0x34: "zh", 0x35: "v", 0x36: "sh", 0x37: "sh",
              0x38: "s", 0x39: "h", 0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
              0x5D: "rh", 0x5E: "f", 0x5F: "y", 0x70: "r", 0x71: "v"}
VOWEL_SIGNS = {0x3A: "e", 0x3B: "e", 0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u",
               0x43: "ri", 0x44: "ri", 0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o",
               0x4A: "o", 0x4B: "o", 0x4C: "au", 0x4F: "au", 0x55: "", 0x56: "ai", 0x57: "au",
               0x62: "li", 0x63: "li"}
NASALS = {0x01, 0x02}            # candrabindu, anusvara (Gurmukhi tippi 0x70 handled below)
VIRAMA = 0x4D
NUKTA = 0x3C
WITH_NUKTA = {"j": "z", "ph": "f", "d": "r", "dh": "rh", "k": "k", "kh": "kh", "g": "g"}
# script blocks in Unicode order
DEVANAGARI, BENGALI, GURMUKHI, GUJARATI, ORIYA, TAMIL, TELUGU, KANNADA, MALAYALAM = range(9)
# Malayalam "chillu" letters and Bengali khanda ta: consonants that never carry a vowel
DEAD_CONSONANTS = {(MALAYALAM, 0x7A): "n", (MALAYALAM, 0x7B): "n", (MALAYALAM, 0x7C): "r",
                   (MALAYALAM, 0x7D): "l", (MALAYALAM, 0x7E): "l", (MALAYALAM, 0x7F): "k",
                   (MALAYALAM, 0x54): "m", (MALAYALAM, 0x55): "y", (MALAYALAM, 0x56): "zh",
                   (MALAYALAM, 0x4E): "r", (BENGALI, 0x4E): "t"}
LABIALS = ("p", "b", "m")
SONORANTS = {"y", "r", "v", "n", "m", "l"}
JOINERS = {"‌", "‍"}


def is_indic(ch):
    return FIRST <= ord(ch) <= LAST


def units_of_word(word):
    """Split an Indian-script word into units:
    ["C", consonant, vowel or None (= not written)] / ["V", vowel] / ["N"] nasal / ["T", text]."""
    units = []
    for ch in word:
        if ch in JOINERS:
            continue
        if not is_indic(ch):
            units.append(["T", ch])
            continue
        block, pos = (ord(ch) - FIRST) // 0x80, (ord(ch) - FIRST) % 0x80
        last = units[-1] if units else None
        if (block, pos) in DEAD_CONSONANTS:
            units.append(["C", DEAD_CONSONANTS[(block, pos)], ""])
        elif block == GURMUKHI and pos == 0x70 or pos in NASALS:   # tippi = nasal
            units.append(["N"])
        elif block == GURMUKHI and pos == 0x71:                    # addak: doubles the next letter
            continue
        elif pos in CONSONANTS:
            units.append(["C", CONSONANTS[pos], None])
        elif pos in VOWELS:
            units.append(["V", VOWELS[pos]])
        elif pos in VOWEL_SIGNS and last and last[0] == "C" and last[2] is None:
            last[2] = VOWEL_SIGNS[pos]
        elif pos in VOWEL_SIGNS:
            units.append(["V", VOWEL_SIGNS[pos]])
        elif pos == VIRAMA and last and last[0] == "C":
            last[2] = ""
        elif pos == NUKTA and last and last[0] == "C":
            last[1] = WITH_NUKTA.get(last[1], last[1])
        elif pos == 0x03:                                          # visarga
            units.append(["T", "h"])
        elif 0x66 <= pos <= 0x6F:                                  # digits
            units.append(["T", str(pos - 0x66)])
        elif pos in (0x64, 0x65):                                  # danda = full stop
            units.append(["T", " "])
        # anything else (accents, signs) carries no sound we need
    return units


def has_vowel(unit):
    """Vowel written, or (not decided yet, to the left) the inherent "a"."""
    return unit[0] == "V" or (unit[0] == "C" and unit[2] != "")


def resolve_inherent_vowels(units, keep_all=False):
    """Decide for every consonant without written vowel: "a" or nothing.
    keep_all: South Indian script, every such consonant keeps its "a"."""
    letters = [i for i, u in enumerate(units) if u[0] in "CV"]
    if not letters:
        return
    if keep_all:
        for i in letters:
            if units[i][0] == "C" and units[i][2] is None:
                units[i][2] = "a"
        return
    for k in range(len(letters) - 1, -1, -1):
        i = letters[k]
        unit = units[i]
        if unit[0] != "C" or unit[2] is not None:
            continue
        nasal_after = i + 1 < len(units) and units[i + 1][0] == "N"
        if nasal_after or len(letters) == 1:
            unit[2] = "a"
        elif k == len(letters) - 1:                                 # end of word
            before = units[letters[k - 1]]
            after_cluster = before[0] == "C" and before[2] == "" and letters[k - 1] == i - 1
            unit[2] = "a" if after_cluster and unit[1] in SONORANTS else ""
        elif k == 0:
            unit[2] = "a"
        else:
            before, after = units[letters[k - 1]], units[letters[k + 1]]
            after_is_cv = after[0] == "C" and bool(after[2])     # decided already (right to left)
            unit[2] = "" if has_vowel(before) and after_is_cv and letters[k - 1] == i - 1 else "a"


def word_to_latin(word):
    malayalam = any(FIRST + 0x80 * MALAYALAM <= ord(ch) < FIRST + 0x80 * (MALAYALAM + 1) for ch in word)
    first = next(ord(ch) for ch in word if is_indic(ch))
    units = units_of_word(word)
    resolve_inherent_vowels(units, keep_all=(first - FIRST) // 0x80 >= TAMIL)
    out = []
    for i, unit in enumerate(units):
        if unit[0] == "C":
            out.append(unit[1] + unit[2])
        elif unit[0] == "V":
            out.append(unit[1])
        elif unit[0] == "N":
            nxt = next((u for u in units[i + 1:] if u[0] in "CV"), None)
            out.append("m" if nxt and nxt[0] == "C" and nxt[1].startswith(LABIALS) else "n")
        else:
            out.append(unit[1])
    text = "".join(out)
    return text.replace("rr", "tt") if malayalam else text   # Malayalam: double r is spoken "tt"


def indic_to_latin(text):
    """Transliterate every Indian-script word of text; other words stay as they are."""
    if not any(is_indic(ch) for ch in text):
        return text
    return " ".join(word_to_latin(w) if any(is_indic(ch) for ch in w) else w for w in text.split())

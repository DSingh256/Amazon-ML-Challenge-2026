"""
Word lists used to clean names and addresses.

Everything here is plain text knowledge (like "rd" means "road").
Nothing here depends on the country of a record, so it works the same
for US, India, France or any new country.
"""

# ---------------------------------------------------------------------------
# Legal words: parts of a business name that say WHAT KIND of company it is,
# not WHICH company it is. We cut them out of the name and keep them aside.
# Each spelling is mapped to one standard short form.
# ---------------------------------------------------------------------------
LEGAL_WORDS = {
    # English / US
    "inc": "inc", "incorporated": "inc",
    "llc": "llc", "llp": "llp", "lp": "lp", "pllc": "pllc", "pc": "pc", "plc": "plc",
    "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co",
    "ltd": "ltd", "limited": "ltd",
    # India
    "pvt": "pvt", "private": "pvt",
    # the same words written in Indian scripts, as indic.py spells them
    "praivet": "pvt", "praibhet": "pvt", "piraivet": "pvt", "praivatt": "pvt",
    "limitet": "ltd", "limittad": "ltd", "limtid": "ltd",
    "elelpi": "llp", "elaelpi": "llp",
    # France / Europe
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "eurl": "eurl", "sa": "sa",
    "sci": "sci", "snc": "snc", "gmbh": "gmbh",
}

# Short words in the middle of dotted abbreviations like "L.L.C." become
# single letters after cleaning ("l l c"). We glue them back first.
LETTER_SEQUENCES = {
    "l l c": "llc", "l l p": "llp", "p l c": "plc",
    "pra li": "pvt ltd",                           # "प्रा. लि." = Pvt. Ltd.
}

# ---------------------------------------------------------------------------
# Address words: short form -> long form.
# Note: some short forms are also state codes ("fl" = floor or Florida,
# "ct" = court or Connecticut). That is fine because we turn state names
# into codes FIRST and expand short forms AFTER, so "Florida", "FL" and "Fl"
# all end up as the same word on both sides of a pair.
# ---------------------------------------------------------------------------
ADDRESS_WORDS = {
    "st": "street", "str": "street",
    "rd": "road",
    "ave": "avenue", "av": "avenue",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "blvd": "boulevard", "bd": "boulevard", "bld": "boulevard",
    "pl": "place",
    "hwy": "highway",
    "pkwy": "parkway",
    "cir": "circle",
    "ter": "terrace",
    "trl": "trail",
    "sq": "square",
    "apt": "apartment", "apts": "apartments",
    "ste": "suite",
    "fl": "floor", "flr": "floor",
    "bldg": "building",
    "opp": "opposite",
    "nr": "near",
    "mkt": "market",
    "ngr": "nagar",
    "r": "rue",            # French: "12 R. de Dieppe"
    "imp": "impasse",
    "ch": "chemin",
    "mt": "mount",
    "ft": "fort",
    "n": "north", "s": "south", "e": "east", "w": "west",
}

# Words that carry no information inside an address. We drop them.
ADDRESS_FILLER_WORDS = {
    "null", "none", "nan", "no", "number", "h", "hno",
}

# ---------------------------------------------------------------------------
# States / regions: full name -> short code.
# Different sources write "North Carolina" or "NC", "Maharashtra" or "MH".
# Mapping both to the same code makes them match.
# ---------------------------------------------------------------------------
STATE_NAMES = {
    # US
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "florida": "fl", "georgia": "ga", "hawaii": "hi", "idaho": "id",
    "illinois": "il", "indiana": "in", "iowa": "ia", "kansas": "ks",
    "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
    "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms",
    "missouri": "mo", "montana": "mt", "nebraska": "ne", "nevada": "nv",
    "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm", "new york": "ny",
    "north carolina": "nc", "north dakota": "nd", "ohio": "oh", "oklahoma": "ok",
    "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
    "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut",
    "vermont": "vt", "virginia": "va", "washington": "wa", "west virginia": "wv",
    "wisconsin": "wi", "wyoming": "wy", "district of columbia": "dc",
    # India (plus the way anyascii writes some of them from Hindi script)
    "andhra pradesh": "ap", "arunachal pradesh": "ar", "assam": "as", "bihar": "br",
    "chhattisgarh": "cg", "goa": "ga", "gujarat": "gj", "haryana": "hr",
    "himachal pradesh": "hp", "jharkhand": "jh", "karnataka": "ka", "krnatk": "ka",
    "kerala": "kl", "madhya pradesh": "mp", "maharashtra": "mh", "mharastr": "mh",
    "manipur": "mn", "meghalaya": "ml", "mizoram": "mz", "nagaland": "nl",
    "odisha": "od", "orissa": "od", "punjab": "pb", "rajasthan": "rj",
    "sikkim": "sk", "tamil nadu": "tn", "telangana": "ts", "tripura": "tr",
    "uttar pradesh": "up", "uttarakhand": "uk", "west bengal": "wb",
    "delhi": "dl", "dilli": "dl", "chandigarh": "ch", "puducherry": "py",
    "jammu and kashmir": "jk", "ladakh": "la",
    # as indic.py writes them from Indian scripts
    "maharashtr": "mh", "karnatak": "ka", "tamizhnatu": "tn", "tamilnatu": "tn", "gujrat": "gj",
    "pashchimbang": "wb", "pashchim bang": "wb", "telangan": "ts", "hariyana": "hr",
    "kerlan": "kl", "keral": "kl", "madhy pradesh": "mp", "andhrapradesh": "ap", "andhr pradesh": "ap",
    "panjab": "pb", "orisha": "od", "odisha": "od", "uttrakhand": "uk", "chhattisgarh": "cg",
    "jharkhand": "jh", "assam": "as", "gova": "ga",
}

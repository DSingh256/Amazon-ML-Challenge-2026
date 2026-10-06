"""
S1 unit tests on real noisy examples from the data.

Run:  python tests/test_normalize.py      (or: pytest tests/)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from normalize import clean_address, clean_name  # noqa: E402

# raw name -> (name_core, legal_form)
NAMES = {
    "-- Holloway Peak Inc Seafood": ("holloway peak seafood", "inc"),
    "Holloway Peak Seafood, Inc.": ("holloway peak seafood", "inc"),
    "RAM MARKETING PVT. LTD.": ("ram marketing", "ltd pvt"),
    "Ram Marketing Private Limited": ("ram marketing", "ltd pvt"),
    "राम मार्केटिंग प्राइवेट लिमिटेड": ("ram marketing", "ltd pvt"),
    "शिवम वेंचर्स प्राइवेट लिमिटेड": ("shivam venchars", "ltd pvt"),
    "कृष्ण सूर्य ट्रेडर्स": ("krishna surya tredars", ""),
    "ಕೃಷ್ಣ ಸೂರ್ಯ ಕನ್ಸಲ್ಟೆಂಟ್ಸ್": ("krishna surya kansaltents", ""),
    "ಮಹೇಶ್ ಎಂಟರ್‌ಪ್ರೈಸಸ್": ("mahesh entarpraisas", ""),
    "L.L.C. Moncada Trucking": ("moncada trucking", "llc"),
    "Moncada Trucking LLC": ("moncada trucking", "llc"),
    "Pvt. EFS Logistics Ltd.": ("efs logistics", "ltd pvt"),
    "www.wilfordhancock.com": ("wilfordhancock", ""),
    "Lord's Bakery & Cafe": ("lords bakery and cafe", ""),
    "Pr0perties Group Corp": ("properties group", "corp"),
    "M/S Shree Ganesh Traders": ("shree ganesh traders", ""),
    "<< Beth Chapel >>": ("beth chapel", ""),
    "Café Élite SARL": ("cafe elite", "sarl"),
    "LLC": ("llc", "llc"),
}

# pairs of raw addresses that must clean to the same text
SAME_ADDRESS = [
    ("01018 KENWOOD ST, HAMMOND, IN", "1018 Kenwood St, Hammond, Indiana"),
    ("12 MG Rd, Bengaluru, Karnataka 560001", "12 M.G. Road, Bengaluru, KA 560001"),
]


def test_names():
    for raw, expected in NAMES.items():
        _, core, legal = clean_name(raw)
        assert (core, legal) == expected, f"{raw!r}: got {(core, legal)}, expected {expected}"


def test_addresses():
    for left, right in SAME_ADDRESS:
        assert clean_address(left) == clean_address(right), (clean_address(left), clean_address(right))


def test_empty():
    assert clean_name("") == ("", "", "")
    assert clean_address("") == ""


if __name__ == "__main__":
    test_names()
    test_addresses()
    test_empty()
    print(f"all S1 tests passed ({len(NAMES)} names, {len(SAME_ADDRESS)} address pairs)")

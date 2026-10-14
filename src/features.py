"""
Shared feature engineering for the MHT-CET college predictor.

These helpers turn raw CAP cutoff rows into model-ready features:
  - region           : extracted from the college name (Mumbai, Pune, Nagpur, ...)
  - gender_scope     : "Female-only" if the category is a Ladies-reserved code (starts with L), else "All"
  - allocation_level : Home-University / Other-than-Home-University / State-level, from the category suffix
  - eligible_category_prefixes : which raw category prefixes a student of a given gender can match against
"""

import re

MH_PLACES = [
    "Mumbai", "Navi Mumbai", "Thane", "Pune", "Nagpur", "Nashik", "Nasik",
    "Aurangabad", "Chhatrapati Sambhajinagar", "Solapur", "Kolhapur", "Amravati",
    "Nanded", "Sangli", "Jalgaon", "Akola", "Latur", "Dhule", "Ahmednagar",
    "Chandrapur", "Parbhani", "Ichalkaranji", "Jalna", "Ambarnath", "Bhusawal",
    "Panvel", "Satara", "Beed", "Yavatmal", "Kamptee", "Gondia", "Barshi",
    "Achalpur", "Osmanabad", "Dharmabad", "Nandurbar", "Wardha", "Udgir",
    "Hinganghat", "Palghar", "Vasai", "Virar", "Mira Bhayandar", "Kalyan",
    "Dombivli", "Ulhasnagar", "Ambernath", "Badlapur", "Ratnagiri", "Sindhudurg",
    "Raigad", "Alibag", "Alibaug", "Karjat", "Khalapur", "Pen", "Roha",
    "Mahad", "Shrirampur", "Sinnar", "Malegaon", "Buldhana", "Hingoli",
    "Washim", "Gadchiroli", "Bhandara", "Wani", "Pandharpur", "Karad",
    "Phaltan", "Baramati", "Pimpri", "Chinchwad", "Lonavala", "Khopoli",
]

_ALIASES = {"Nasik": "Nashik", "Alibag": "Alibaug"}

_PLACE_RE = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(MH_PLACES, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)


def extract_region(college_name: str) -> str:
    """Pull a Maharashtra place name out of the college name string."""
    if not isinstance(college_name, str):
        return "Unknown"
    match = _PLACE_RE.search(college_name)
    if not match:
        return "Unknown"
    found = match.group(1)
    # normalize capitalization to match our canonical list
    for place in MH_PLACES:
        if place.lower() == found.lower():
            return _ALIASES.get(place, place)
    return _ALIASES.get(found, found)


def gender_scope(category: str) -> str:
    """Ladies-reserved category codes start with 'L' (e.g. LOPENH, LSCH)."""
    if not isinstance(category, str) or len(category) == 0:
        return "All"
    return "Female-only" if category.upper().startswith("L") else "All"


def allocation_level(category: str) -> str:
    """
    The trailing letter of most MHT-CET category codes marks the allocation scope:
      H -> Home University
      O -> Other-than-Home University
      S -> State level
    Falls back to 'State' when no suffix is present (open/general all-state pools).
    """
    if not isinstance(category, str) or len(category) == 0:
        return "State"
    c = category.upper()
    if c.endswith("H"):
        return "Home University"
    if c.endswith("O"):
        return "Other University"
    if c.endswith("S"):
        return "State"
    return "State"


def eligible_category_prefixes(gender: str) -> list:
    """
    Which raw-category family prefixes a student of this gender can be matched
    against. Female students are eligible for both Ladies-only (L) and general (G/other)
    seats; everyone else is eligible for general seats only.
    """
    if isinstance(gender, str) and gender.strip().lower() in ("female", "f", "woman", "women"):
        return ["L", "G", "_OTHER_"]
    return ["G", "_OTHER_"]

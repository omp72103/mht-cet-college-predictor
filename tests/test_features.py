from src.features import allocation_level, eligible_category_prefixes, extract_region, gender_scope


def test_extract_region_and_aliases():
    assert extract_region("College of Engineering, Pune") == "Pune"
    assert extract_region("XYZ Institute, Nasik") == "Nashik"
    assert extract_region("Nowhere Institute") == "Unknown"
    assert extract_region(None) == "Unknown"


def test_gender_scope():
    assert gender_scope("LOPENH") == "Female-only"
    assert gender_scope("GOPENS") == "All"
    assert gender_scope("") == "All"


def test_allocation_level():
    assert allocation_level("GOPENH") == "Home University"
    assert allocation_level("GOPENO") == "Other University"
    assert allocation_level("GOPENS") == "State"
    assert allocation_level(None) == "State"


def test_eligible_prefixes():
    assert eligible_category_prefixes("Female") == ["L", "G", "_OTHER_"]
    assert eligible_category_prefixes("Male") == ["G", "_OTHER_"]

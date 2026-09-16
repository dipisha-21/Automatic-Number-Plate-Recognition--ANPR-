from src.anpr.text import normalize_plate, levenshtein_distance, character_accuracy


def test_normalize_plate():
    assert normalize_plate(" mh-12 ab 1234 ") == "MH12AB1234"


def test_levenshtein_distance():
    assert levenshtein_distance("MH12AB1234", "MH12AB1234") == 0
    assert levenshtein_distance("MH12AB1234", "MH12AB1235") == 1


def test_character_accuracy():
    assert character_accuracy("MH12AB1234", "MH12AB1234") == 1.0
    assert 0 < character_accuracy("MH12AB1235", "MH12AB1234") < 1

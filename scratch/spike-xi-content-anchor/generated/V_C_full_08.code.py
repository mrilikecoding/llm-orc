import pytest
from text_tools import squeeze_runs, tally_class

def test_squeeze_runs():
    test_cases = [
        ("a  b", " ", "a b"),
        ("   ", " ", " "),
        ("a!!b", "!", "a!b"),
        ("abc", " ", "abc"),
        ("a   b c", " ", "a b c"),
        ("123", "0123", "1"),
    ]
    for input_text, chars, expected in test_cases:
        assert squeeze_runs(input_text, chars=chars) == expected

def test_tally_class():
    test_cases = [
        ("a1b2c3", "digit", 3),
        ("a1b2c3", "alpha", 3),
        ("a b c", "space", 2),
        ("123abc", "digit", 3),
        ("123abc", "alpha", 3),
        ("   ", "space", 3),
    ]
    for text, predicate, expected in test_cases:
        assert tally_class(text, predicate_name=predicate) == expected

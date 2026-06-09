import text_tools
import pytest


def test_squeeze_runs_normal():
    assert text_tools.squeeze_runs("hello   world") == "hello world"


def test_squeeze_runs_multiple_chars():
    assert text_tools.squeeze_runs("xxxyyyzzz", chars="xyz") == "xyz"


def test_squeeze_runs_empty():
    assert text_tools.squeeze_runs("") == ""


def test_squeeze_runs_all_chars():
    assert text_tools.squeeze_runs("   ", chars=" ") == " "


def test_tally_class_digits():
    assert text_tools.tally_class("abc123", "digit") == 3


def test_tally_class_alpha():
    assert text_tools.tally_class("abc123", "alpha") == 3


def test_tally_class_space():
    assert text_tools.tally_class("a b c", "space") == 2


def test_tally_class_invalid_predicate():
    with pytest.raises(KeyError):
        text_tools.tally_class("abc123", "punctuation")

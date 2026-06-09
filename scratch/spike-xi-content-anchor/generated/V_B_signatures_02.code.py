import pytest
from text_tools import squeeze_runs, tally_class

def test_squeeze_runs_basic():
    assert squeeze_runs("hello   world") == "hello world"

def test_squeeze_runs_other_chars():
    assert squeeze_runs("a..b..", chars=".") == "a.b."

def test_squeeze_runs_default_chars():
    assert squeeze_runs("a  b  c") == "a b c"

def test_tally_class_digits():
    assert tally_class("abc123", "digit") == 3

def test_tally_class_alpha():
    assert tally_class("abc123", "alpha") == 3

def test_tally_class_spaces():
    assert tally_class("hello world", "space") == 1

def test_tally_class_empty():
    assert tally_class("", "digit") == 0

def test_tally_class_no_matches():
    assert tally_class("abc123", "space") == 0

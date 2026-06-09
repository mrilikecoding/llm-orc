import pytest
from text_tools import squeeze_runs, tally_class

def test_squeeze_runs_default():
    assert squeeze_runs("a  b c") == "a b c"

def test_squeeze_runs_custom_chars():
    assert squeeze_runs("x.x.x", chars="x") == "x.x"

def test_squeeze_runs_no_change():
    assert squeeze_runs("abc") == "abc"

def test_tally_class_digits():
    assert tally_class("abc123", predicate_name="digit") == 3

def test_tally_class_alpha():
    assert tally_class("abc123", predicate_name="alpha") == 3

def test_tally_class_spaces():
    assert tally_class("a b c", predicate_name="space") == 2

def test_tally_class_mixed():
    assert tally_class("a1!b", predicate_name="digit") == 1
    assert tally_class("a1!b", predicate_name="alpha") == 2
    assert tally_class("a1!b", predicate_name="space") == 0

import text_tools


def test_squeeze_runs_basic():
    assert text_tools.squeeze_runs("   ") == " "


def test_squeeze_runs_multiple():
    assert text_tools.squeeze_runs("a  b") == "a b"


def test_squeeze_runs_mixed_chars():
    assert text_tools.squeeze_runs("aaabbbccc", chars="abc") == "abc"


def test_squeeze_runs_other_chars():
    assert text_tools.squeeze_runs("hello,,world", chars=",") == "hello,world"


def test_squeeze_runs_no_change():
    assert text_tools.squeeze_runs("a b c", chars=" ") == "a b c"


def test_squeeze_runs_empty():
    assert text_tools.squeeze_runs("") == ""


def test_tally_class_digit():
    assert text_tools.tally_class("123abc", "digit") == 3


def test_tally_class_alpha():
    assert text_tools.tally_class("123abc", "alpha") == 3


def test_tally_class_space():
    assert text_tools.tally_class("123\n\tabc", "space") == 2

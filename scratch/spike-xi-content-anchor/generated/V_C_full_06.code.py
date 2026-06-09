import text_tools


def test_squeeze_runs():
    assert text_tools.squeeze_runs("a  b  c", chars=" ") == "a b c"
    assert text_tools.squeeze_runs("Hello   World", chars=" ") == "Hello World"
    assert text_tools.squeeze_runs("", chars=" ") == ""
    assert text_tools.squeeze_runs("-----", chars="-") == "-"
    assert text_tools.squeeze_runs("x--y--z", chars="-") == "x-y-z"


def test_tally_class():
    assert text_tools.tally_class("abc123", predicate_name="alpha") == 3
    assert text_tools.tally_class("abc123", predicate_name="digit") == 3
    assert text_tools.tally_class(" 123 ", predicate_name="space") == 2
    assert text_tools.tally_class("A1!@#", predicate_name="alpha") == 1
    assert text_tools.tally_class("A1!@#", predicate_name="digit") == 1

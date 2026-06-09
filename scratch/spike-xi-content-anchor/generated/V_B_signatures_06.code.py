import text_tools

def test_squeeze_runs():
    assert text_tools.squeeze_runs("Hello   world") == "Hello world"
    assert text_tools.squeeze_runs("a--b--c", chars="-") == "a-b-c"
    assert text_tools.squeeze_runs("a,,b,,c", chars=",") == "a,b,c"

def test_tally_class():
    assert text_tools.tally_class("abc123", "digit") == 3
    assert text_tools.tally_class("abc", "digit") == 0
    assert text_tools.tally_class("abc123", "alpha") == 3
    assert text_tools.tally_class("123", "alpha") == 0
    assert text_tools.tally_class("a b c", "space") == 2
    assert text_tools.tally_class("hello", "space") == 0

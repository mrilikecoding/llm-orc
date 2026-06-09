import text_tools

def test_squeeze_runs():
    assert text_tools.squeeze_runs("a  b") == "a b"
    assert text_tools.squeeze_runs("a--b", chars="-") == "a-b"
    assert text_tools.squeeze_runs("   ") == " "
    assert text_tools.squeeze_runs("no spaces here") == "no spaces here"

def test_tally_class():
    assert text_tools.tally_class("abc123def456", "digit") == 6
    assert text_tools.tally_class("abc123def456", "alpha") == 6
    assert text_tools.tally_class("a b c", "space") == 2
    assert text_tools.tally_class("abc123", "invalid") == 0

import text_tools

def test_squeeze_runs_default():
    assert text_tools.squeeze_runs("hello   world") == "hello world"

def test_squeeze_runs_custom_chars():
    assert text_tools.squeeze_runs("a--b++c", chars="-+") == "a-b+c"

def test_tally_class_digit():
    assert text_tools.tally_class("abc123", "digit") == 3

def test_tally_class_alpha():
    assert text_tools.tally_class("abc123", "alpha") == 3

def test_tally_class_space():
    assert text_tools.tally_class("a b c", "space") == 2

def test_tally_class_digit_no_digits():
    assert text_tools.tally_class("abc", "digit") == 0

def test_tally_class_alpha_no_alpha():
    assert text_tools.tally_class("123", "alpha") == 0

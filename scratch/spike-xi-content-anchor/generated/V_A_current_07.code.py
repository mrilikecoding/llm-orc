import text_tools

def test_clean_text():
    assert text_tools.clean_text("  Hello, World!  ") == "hello world"
    assert text_tools.clean_text("   ") == ""
    assert text_tools.clean_text("Mixed CASE! 123") == "mixed case 123"

def test_count_words():
    assert text_tools.count_words("Hello world") == 2
    assert text_tools.count_words("   ") == 0
    assert text_tools.count_punctuation("one, two; three") == 3

def test_remove_punctuation():
    assert text_tools.remove_punctuation("Hello, World!") == "Hello World"
    assert text_tools.remove_punctuation("Test: 123") == "Test 123"
    assert text_tools.remove_punctuation("") == ""

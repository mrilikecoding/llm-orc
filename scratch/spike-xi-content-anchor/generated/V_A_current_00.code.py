import text_tools

def test_clean_text():
    assert text_tools.clean_text("  Hello, World!  ") == "hello world"
    assert text_tools.clean_text("   ") == ""
    assert text_tools.clean_text("ALL CAPS") == "all caps"

def test_remove_stopwords():
    assert text_tools.remove_stopwords(["the", "quick", "brown", "fox"]) == ["quick", "brown", "fox"]
    assert text_tools.remove_stopwords(["apple", "banana"]) == ["apple", "banana"]
    assert text_tools.remove_stopwords([]) == []

def test_tokenize_text():
    assert text_tools.tokenize_text("Hello, world!") == ["hello", "world"]
    assert text_tools.tokenize_text("   Split   this   ") == ["split", "this"]
    assert text_tools.tokenize_text("") == []

# test_text_tools.py
import pytest
from text_tools import *  # Replace with actual function imports if known


def test_reverse_text():
    """Test the reverse_text function."""
    assert reverse_text("hello") == "olleh"
    assert reverse_text(" ") == " "
    assert reverse_text("a b c") == "c b a"


def test_clean_text():
    """Test the clean_text function."""
    assert clean_text("  Hello   world!  ") == "hello world!"
    assert clean_text("  123   ") == "123"
    assert clean  ("  MIXED CASE  ") == "mixed case"


def test_count_words():
    """Test the count_words function."""
    assert count_words("hello world") == 2
    assert count_words("   ") == 0
    assert count_words("one") == 1
    assert count_words("a b c d") == 4


def test_remove_stopwords():
    """Test the remove_stopwords function."""
    assert remove_stopwords("the cat sat on the mat") == "cat sat mat"
    assert remove_stopwords("a a a") == ""
    assert remove_stopwords("hello world") == "hello world"


def test_tokenize_text():
    """Test the tokenize_text function."""
    assert tokenize_text("Hello, world!") == ["Hello", "world"]
    assert tokenize_text("  split   these   words  ") == ["split", "these", "words"]
    assert tokenize_text("no-underscore") == ["no", "underscore"]

import pytest
import text_tools

def test_clean_text():
    """Test that clean_text removes punctuation and lowercases text."""
    assert text_tools.clean_text("Hello, World!") == "hello world"
    assert text_tools.clean_text("  Test   ") == "test"
    assert text_tools.clean_text("NoChangesHere") == "nochangeshere"

def test_split_words():
    """Test that split_words splits text into lowercase words."""
    assert text_tools.split_words("Hello world!") == ["hello", "world"]
    assert text_tools.split_words("   Multiple    spaces   ") == ["multiple", "spaces"]
    assert text_tools.split_words("Punctuation, here!") == ["punctuation", "here"]

def test_remove_extra_spaces():
    """Test that remove_extra_spaces trims and collapses multiple spaces."""
    assert text_tools.remove_extra_spaces("  Leading  spaces  ") == "Leading spaces"
    assert text_tools.remove_extra_spaces("No   extra   spaces") == "No extra spaces"
    assert text_tools.remove_punctuation("Text with  punctuation!") == "text with punctuation"

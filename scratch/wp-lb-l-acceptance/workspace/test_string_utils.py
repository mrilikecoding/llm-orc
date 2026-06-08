# test_string_utils.py
import unittest
from string_utils import reverse_word_order

class TestReverseWordOrder(unittest.TestCase):
    def test_reverse_words(self):
        self.assertEqual(reverse_word_order("hello world"), "world hello")
        self.assertEqual(reverse_word_order("   leading spaces"), "spaces leading   ")
        self.assertEqual(reverse_word_order("single"), "single")
        self.assertEqual(reverse_word_order(""), "")
        self.assertEqual(reverse_word_order("multiple   spaces"), "spaces   multiple")
        self.assertEqual(reverse_word_order("   middle   words   "), "   words   middle   ")

if __name__ == '__main__':
    unittest.main()
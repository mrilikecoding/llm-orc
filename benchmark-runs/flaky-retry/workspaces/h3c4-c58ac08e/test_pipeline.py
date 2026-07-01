import unittest
from pipeline import run
from errors import ParseError, RangeError

class TestPipeline(unittest.TestCase):
    def test_valid_input(self):
        self.assertEqual(run("42", 0, 100), 42)

    def test_parse_error(self):
        with self.assertRaises(ParseError):
            run("abc", 0, 100)

    def test_range_error(self):
        with self.assertRaises(RangeError):
            run("50", 0, 10)

if __name__ == '__main__':
    unittest.main()
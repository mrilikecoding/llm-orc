import unittest
from pipeline import run
from errors import ParseError, RangeError

class TestPipeline(unittest.TestCase):
    def test_happy_path(self):
        self.assertEqual(run("5", 1, 10), 5)

    def test_parse_error(self):
        with self.assertRaises(ParseError):
            run("abc", 1, 10)

    def test_range_error(self):
        with self.assertRaises(RangeError):
            run("50", 1, 10)

if __name__ == '__main__':
    unittest.main()
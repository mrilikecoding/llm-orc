import unittest
from pipeline import run

class TestPipeline(unittest.TestCase):
    def test_success(self):
        result = run("123", 100, 200)
        self.assertEqual(result, 123)

    def test_parse_error(self):
        with self.assertRaises(ParseError):
            run("abc", 100, 200)

    def test_range_error_low(self):
        with self.assertRaises(RangeError):
            run("50", 100, 200)

    def test_range_error_high(self):
        with self.assertRaises(RangeError):
            run("200", 100, 150)

if __name__ == '__main__':
    unittest.main()
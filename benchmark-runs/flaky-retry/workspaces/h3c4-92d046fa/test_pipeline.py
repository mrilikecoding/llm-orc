import unittest
from pipeline import run

class TestPipeline(unittest.TestCase):
    def test_parse_error(self):
        with self.assertRaises(ParseError):
            run("abc", 1, 10)

    def test_range_error(self):
        with self.assertRaises(RangeError):
            run("5", 10, 20)

    def test_valid_input(self):
        result = run("15", 10, 20)
        self.assertEqual(result, 15)

if __name__ == '__main__':
    unittest.main()
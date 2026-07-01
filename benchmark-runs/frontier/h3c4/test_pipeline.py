import unittest

from pipeline import run
from errors import ParseError, RangeError


class TestPipeline(unittest.TestCase):
    def test_valid_input_returns_int(self) -> None:
        self.assertEqual(run("5", 1, 10), 5)

    def test_non_numeric_raises_parse_error(self) -> None:
        with self.assertRaises(ParseError):
            run("abc", 1, 10)

    def test_below_range_raises_range_error(self) -> None:
        with self.assertRaises(RangeError):
            run("0", 1, 10)

    def test_above_range_raises_range_error(self) -> None:
        with self.assertRaises(RangeError):
            run("11", 1, 10)

    def test_boundary_low_accepted(self) -> None:
        self.assertEqual(run("1", 1, 10), 1)

    def test_boundary_high_accepted(self) -> None:
        self.assertEqual(run("10", 1, 10), 10)


if __name__ == "__main__":
    unittest.main()

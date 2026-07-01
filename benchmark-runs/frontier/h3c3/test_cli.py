import io
import sys
import unittest
from unittest.mock import patch

from cli import main


class TestCLIToFahrenheit(unittest.TestCase):
    def test_zero_celsius_to_fahrenheit(self) -> None:
        with patch("sys.argv", ["cli", "0", "--to-fahrenheit"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                main()
        self.assertAlmostEqual(float(captured.getvalue().strip()), 32.0)

    def test_100_celsius_to_fahrenheit(self) -> None:
        with patch("sys.argv", ["cli", "100", "--to-fahrenheit"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                main()
        self.assertAlmostEqual(float(captured.getvalue().strip()), 212.0)


class TestCLIToKelvin(unittest.TestCase):
    def test_zero_celsius_to_kelvin(self) -> None:
        with patch("sys.argv", ["cli", "0", "--to-kelvin"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                main()
        self.assertAlmostEqual(float(captured.getvalue().strip()), 273.15)

    def test_100_celsius_to_kelvin(self) -> None:
        with patch("sys.argv", ["cli", "100", "--to-kelvin"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                main()
        self.assertAlmostEqual(float(captured.getvalue().strip()), 373.15)


class TestCLIToCelsius(unittest.TestCase):
    def test_32_fahrenheit_to_celsius(self) -> None:
        with patch("sys.argv", ["cli", "32", "--to-celsius"]):
            captured = io.StringIO()
            with patch("sys.stdout", captured):
                main()
        self.assertAlmostEqual(float(captured.getvalue().strip()), 0.0)


class TestCLIMutualExclusion(unittest.TestCase):
    def test_no_flag_exits(self) -> None:
        with patch("sys.argv", ["cli", "100"]):
            with self.assertRaises(SystemExit):
                main()

    def test_both_flags_exits(self) -> None:
        with patch("sys.argv", ["cli", "100", "--to-fahrenheit", "--to-kelvin"]):
            with self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()

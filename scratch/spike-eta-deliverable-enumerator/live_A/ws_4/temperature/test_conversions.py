

import unittest
from temperature_conversion import (
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    celsius_to_kelvin
)


class TestCelsiusToFahrenheit(unittest.TestCase):
    def test_zero_returns_32(self):
        """Test freezing point of water."""
        self.assertEqual(celsius_to_fahrenheit(0), 32)

    def test_100_returns_212(self):
        """Test boiling point of water."""
        self.assertEqual(celsius_to_fahrenheit(100), 212)

    def test_negative_40(self):
        """Test the -40 intersection point where C = F."""
        self.assertEqual(celsius_to_fahrenheit(-40), -40)


class TestFahrenheitToCelsius(unittest.TestCase):
    def test_32_returns_zero(self):
        """Test freezing point of water."""
        self.assertEqual(fahrenheit_to_celsius(32), 0)

    def test_212_returns_100(self):
        """Test boiling point of water."""
        self.assertEqual(fahrenheit_to_celsius(212), 100)

    def test_negative_40(self):
        """Test the -40 intersection point where F = C."""
        self.assertEqual(fahrenheit_to_celsius(-40), -40)


class TestCelsiusToKelvin(unittest.TestCase):
    def test_zero_returns_273_15(self):
        """Test freezing point of water in Kelvin."""
        self.assertAlmostEqual(celsius_to_kelvin(0), 273.15, places=2)

    def test_100_returns_373_15(self):
        """Test boiling point of water in Kelvin."""
        self.assertAlmostEqual(celsius_to_kelvin(100), 373.15, places=2)

    def test_absolute_zero(self):
        """Test absolute zero."""
        self.assertAlmostEqual(celsius_to_kelvin(-273.15), 0, places=2)


if __name__ == "__main__":
    unittest.main()
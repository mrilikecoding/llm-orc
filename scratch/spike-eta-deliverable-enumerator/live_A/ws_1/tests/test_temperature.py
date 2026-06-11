

import unittest
from temperature import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin


class TestTemperatureConversions(unittest.TestCase):
    def test_celsius_to_fahrenheit_freezing_point(self):
        result = celsius_to_fahrenheit(0)
        self.assertEqual(result, 32.0)

    def test_celsius_to_fahrenheit_boiling_point(self):
        result = celsius_to_fahrenheit(100)
        self.assertEqual(result, 212.0)

    def test_celsius_to_fahrenheit_negative(self):
        result = celsius_to_fahrenheit(-40)
        self.assertEqual(result, -40.0)

    def test_fahrenheit_to_celsius_freezing_point(self):
        result = fahrenheit_to_celsius(32)
        self.assertEqual(result, 0.0)

    def test_fahrenheit_to_celsius_boiling_point(self):
        result = fahrenheit_to_celsius(212)
        self.assertEqual(result, 100.0)

    def test_fahrenheit_to_celsius_negative(self):
        result = fahrenheit_to_celsius(-40)
        self.assertEqual(result, -40.0)

    def test_celsius_to_kelvin_freezing_point(self):
        result = celsius_to_kelvin(0)
        self.assertEqual(result, 273.15)

    def test_celsius_to_kelvin_boiling_point(self):
        result = celsius_to_kelvin(100)
        self.assertEqual(result, 373.15)

    def test_celsius_to_kelvin_absolute_zero(self):
        result = celsius_to_kelvin(-273.15)
        self.assertEqual(result, 0.0)


if __name__ == "__main__":
    unittest.main()
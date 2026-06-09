import unittest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

class TestConverters(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        self.assertAlmostEqual(celsius_to_fahrenheit(0), 32.0)
        self.assertAlmostEqual(celsius_to_fahrenheit(-40), -40.0)
        self.assertAlmostEqual(celsius_to_fahrenheit(100), 212.0)

    def test_fahrenheit_to_celsius(self):
        self.assertAlmostEqual(fahrenheit_to_celsius(32), 0.0)
        self.assertAlmostEqual(fahrenheit_to_celsius(212), 100.0)
        self.assertAlmostEqual(fahrenheit_to_celsius(-40), -40.0)

    def test_celsius_to_kelvin(self):
        self.assertAlmostEqual(celsius_to_kelvin(0), 273.15)
        self.assertAlmostEqual(celsius_to_kelvin(-273.15), 0.0)
        self.assertAlmostEqual(celsius_to_kelvin(100), 373.15)

if __name__ == '__main__':
    unittest.main()

The `celsius_to_kelvin` test case for -273.15°C assumes the function handles negative inputs, which may not be intended if the function is designed to raise an error for absolute zero. Additionally, the tests lack checks for floating-point precision beyond the default delta, which could miss subtle implementation errors.
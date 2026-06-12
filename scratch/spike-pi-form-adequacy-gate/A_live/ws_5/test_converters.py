import unittest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

class TestConverters(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        self.assertAlmostEqual(celsius_to_fahrenheit(0), 32.0)
        self.assertAlmostEqual(celsius_to_fahrenheit(100), 212.0)
        self.assertAlmostEqual(celsius_to_fahrenheit(-40), -40.0)
        self.assertAlmostEqual(celsius_to_fahrenheit(25.5), 77.9)
        self.assertAlmostEqual(celsius_to_fahrenheit(-273.15), -459.67)

    def test_fahrenheit_to_celsius(self):
        self.assertAlmostEqual(fahrenheit_to_celsius(32), 0.0)
        self.assertAlmostEqual(fahrenheit_to_celsius(212), 100.0)
        self.assertAlmostEqual(fahrenheit_to_celsius(230), 110.0)
        self.assertAlmostEqual(fahrenheit_to_celsius(25.5), -3.6111111111)
        self.assertAlmostEqual(fahrenheit_to_celsius(-40), -40.0)

    def test_celsius_to_kelvin(self):
        self.assertAlmostEqual(celsius_to_kelvin(0), 273.15)
        self.assertAlmostEqual(celsius_to_kelvin(100), 373.15)
        self.assertAlmostEqual(celsius_to_kelvin(-273.15), 0.0)
        self.assertAlmostEqual(celsius_to_kelvin(25.5), 298.65)
        self.assertAlmostEqual(celsius_to_kelvin(-10), 263.15)

if __name__ == '__main__':
    unittest.main()
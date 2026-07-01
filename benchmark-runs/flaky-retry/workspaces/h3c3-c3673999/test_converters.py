import unittest
import converters

class TestConverters(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        self.assertAlmostEqual(converters.celsius_to_fahrenheit(0), 32.0)
        self.assertAlmostEqual(converters.celsius_to_fahrenheit(100), 212.0)
        self.assertAlmostEqual(converters.celsius_to_fahrenheit(-40), -40.0)
        self.assertAlmostEqual(converters.celsius_to_fahrenheit(25.5), 77.9)
        self.assertAlmostEqual(converters.celsius_to_fahrenheit(1000), 1832.0)
        self.assertAlmostEqual(converters.celsius_to_fahrenheit(-100), -148.0)

    def test_fahrenheit_to_celsius(self):
        self.assertAlmostEqual(converters.fahrenheit_to_celsius(32), 0.0)
        self.assertAlmostEqual(converters.fahrenheit_to_celsius(212), 100.0)
        self.assertAlmostEqual(converters.fahrenheit_to_celsius(-40), -40.0)
        self.assertAlmostEqual(converters.fahrenheit_to_celsius(77.9), 25.5)
        self.assertAlmostEqual(converters.fahrenheit_to_celsius(0), -17.7777777778)
        self.assertAlmostEqual(converters.fahrenheit_to_celsius(1000), 537.7777777778)

    def test_celsius_to_kelvin(self):
        self.assertAlmostEqual(converters.celsius_to_kelvin(0), 273.15)
        self.assertAlmostEqual(converters.celsius_to_kelvin(100), 373.15)
        self.assertAlmostEqual(converters.celsius_to_kelvin(-273.15), 0.0)
        self.assertAlmostEqual(converters.celsius_to_kelvin(25.5), 298.65)
        self.assertAlmostEqual(converters.celsius_to_kelvin(-1000), -726.85)
        self.assertAlmostEqual(converters.celsius_to_kelvin(-100), 173.15)

if __name__ == '__main__':
    unittest.main()
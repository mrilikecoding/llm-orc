import unittest
import converters

class TestConverters(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        self.assertEqual(converters.celsius_to_fahrenheit(0), 32)
        self.assertEqual(converters.celsius_to_fahrenheit(100), 212)
        self.assertEqual(converters.celsius_to_fahrenheit(-40), -40)
        self.assertEqual(converters.celsius_to_fahrenheit(25), 77.0)

    def test_fahrenheit_to_celsius(self):
        self.assertEqual(converters.fahrenheit_to_celsius(32), 0)
        self.assertEqual(converters.fahrenheit_to_celsius(212), 100)
        self.assertEqual(converters.fahrenheit_to_celsius(-40), -40)
        self.assertEqual(converters.fahrenheit_to_celsius(50), 10.0)

    def test_celsius_to_kelvin(self):
        self.assertEqual(converters.celsius_to_kelvin(0), 273.15)
        self.assertEqual(converters.celsius_to_kelvin(100), 373.15)
        self.assertEqual(converters.celsius_to_kelvin(-273.15), 0.0)
        self.assertEqual(converters.celsius_to_kelvin(25), 298.15)

if __name__ == '__main__':
    unittest.main()
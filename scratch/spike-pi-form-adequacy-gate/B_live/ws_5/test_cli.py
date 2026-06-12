import unittest
import converters

class TestConverters(unittest.TestCase):
    def test_celsius_to_fahrenheit(self):
        self.assertEqual(converters.celsius_to_fahrenheit(32), 89.6)
        self.assertEqual(converters.celsius_to_fahrenheit(0), 32.0)
    
    def test_fahrenheit_to_celsius(self):
        self.assertEqual(converters.fahrenheit_to_celsius(98.6), 37.0)
        self.assertEqual(converters.fahrenheit_to_celsius(212), 100.0)
    
    def test_celsius_to_kelvin(self):
        self.assertEqual(converters.celsius_to_kelvin(0), 273.15)
        self.assertEqual(converters.celsius_to_kelvin(-273.15), 0.0)
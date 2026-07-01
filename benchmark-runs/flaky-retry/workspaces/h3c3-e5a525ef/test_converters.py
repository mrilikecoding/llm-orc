import unittest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

class TestConverters(unittest.TestCase):
    def test_celsius_to_fahrenheit_normal(self):
        self.assertAlmostEqual(celsius_to_fahrenheit(25), 77.0, places=1)
    
    def test_celsius_to_fahrenheit_edge_0(self):
        self.assertEqual(celsius_to_fahrenheit(0), 32.0)
    
    def test_celsius_to_fahrenheit_edge_neg40(self):
        self.assertAlmostEqual(celsius_to_fahrenheit(-40), -40.0, places=1)
    
    def test_celsius_to_fahrenheit_edge_100(self):
        self.assertEqual(celsius_to_fahrenheit(100), 212.0)
    
    def test_fahrenheit_to_celsius_normal(self):
        self.assertAlmostEqual(fahrenheit_to_celsius(70), 21.1111, places=4)
    
    def test_fahrenheit_to_celsius_edge_32(self):
        self.assertEqual(fahrenheit_to_celsius(32), 0.0)
    
    def test_fahrenheit_to_celsius_edge_neg40(self):
        self.assertEqual(fahrenheit_to_celsius(-40), -40.0)
    
    def test_fahrenheit_to_celsius_edge_212(self):
        self.assertEqual(fahrenheit_to_celsius(212), 100.0)
    
    def test_celsius_to_kelvin_normal(self):
        self.assertAlmostEqual(celsius_to_kelvin(25), 298.15, places=2)
    
    def test_celsius_to_kelvin_edge_0(self):
        self.assertEqual(celsius_to_kelvin(0), 273.15)
    
    def test_celsius_to_kelvin_edge_neg40(self):
        self.assertAlmostEqual(celsius_to_kelvin(-40), 233.15, places=2)
    
    def test_celsius_to_kelvin_edge_100(self):
        self.assertEqual(celsius_to_kelvin(100), 373.15)

if __name__ == '__main__':
    unittest.main()
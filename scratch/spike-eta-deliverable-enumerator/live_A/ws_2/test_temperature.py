

import unittest
import temperature


class TestCelsiusToFahrenheit(unittest.TestCase):
    
    def test_freezing_point(self):
        self.assertEqual(temperature.celsius_to_fahrenheit(0), 32.0)
    
    def test_boiling_point(self):
        self.assertEqual(temperature.celsius_to_fahrenheit(100), 212.0)
    
    def test_negative_forty_equal(self):
        self.assertEqual(temperature.celsius_to_fahrenheit(-40), -40.0)
    
    def test_room_temperature(self):
        self.assertAlmostEqual(temperature.celsius_to_fahrenheit(25), 77.0)
    
    def test_negative_temperature(self):
        self.assertAlmostEqual(temperature.celsius_to_fahrenheit(-10), 14.0)


class TestFahrenheitToCelsius(unittest.TestCase):
    
    def test_freezing_point(self):
        self.assertEqual(temperature.fahrenheit_to_celsius(32), 0.0)
    
    def test_boiling_point(self):
        self.assertEqual(temperature.fahrenheit_to_celsius(212), 100.0)
    
    def test_negative_forty_equal(self):
        self.assertEqual(temperature.fahrenheit_to_celsius(-40), -40.0)
    
    def test_room_temperature(self):
        self.assertAlmostEqual(temperature.fahrenheit_to_celsius(77), 25.0)
    
    def test_negative_temperature(self):
        self.assertAlmostEqual(temperature.fahrenheit_to_celsius(14), -10.0)
    
    def test_zero(self):
        self.assertAlmostEqual(temperature.fahrenheit_to_celsius(32), 0.0)


class TestCelsiusToKelvin(unittest.TestCase):
    
    def test_freezing_point(self):
        self.assertAlmostEqual(temperature.celsius_to_kelvin(0), 273.15)
    
    def test_boiling_point(self):
        self.assertAlmostEqual(temperature.celsius_to_kelvin(100), 373.15)
    
    def test_absolute_zero(self):
        self.assertAlmostEqual(temperature.celsius_to_kelvin(-273.15), 0.0)
    
    def test_zero_degrees_celsius(self):
        self.assertAlmostEqual(temperature.celsius_to_kelvin(0), 273.15)
    
    def test_room_temperature(self):
        self.assertAlmostEqual(temperature.celsius_to_kelvin(25), 298.15)
    
    def test_negative_temperature(self):
        self.assertAlmostEqual(temperature.celsius_to_kelvin(-10), 263.15)


class TestRoundTrips(unittest.TestCase):
    
    def test_celsius_fahrenheit_celsius(self):
        original = 25.0
        converted = temperature.celsius_to_fahrenheit(original)
        back = temperature.fahrenheit_to_celsius(converted)
        self.assertAlmostEqual(back, original)
    
    def test_freezing_round_trip(self):
        c = 0.0
        f = temperature.celsius_to_fahrenheit(c)
        c_back = temperature.fahrenheit_to_celsius(f)
        self.assertAlmostEqual(c_back, c)
    
    def test_boiling_round_trip(self):
        c = 100.0
        f = temperature.celsius_to_fahrenheit(c)
        c_back = temperature.fahrenheit_to_celsius(f)
        self.assertAlmostEqual(c_back, c)


class TestEdgeCases(unittest.TestCase):
    
    def test_very_high_temperature(self):
        self.assertAlmostEqual(temperature.celsius_to_fahrenheit(1000), 1832.0)
    
    def test_very_low_temperature(self):
        self.assertAlmostEqual(temperature.celsius_to_fahrenheit(-273.15), -459.67)
    
    def test_decimal_values(self):
        self.assertAlmostEqual(temperature.celsius_to_fahrenheit(36.6), 97.88)
    
    def test_kelvin_absolute_zero_edge(self):
        result = temperature.celsius_to_kelvin(-273.15)
        self.assertGreaterEqual(result, 0)


if __name__ == '__main__':
    unittest.main()
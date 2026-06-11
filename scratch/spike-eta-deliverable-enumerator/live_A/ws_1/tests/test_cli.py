

import unittest
from temperature_cli import convert_temperature


class TestTemperatureConversions(unittest.TestCase):
    def test_celsius_to_fahrenheit_freezing_point(self):
        result = convert_temperature(0, "celsius", "fahrenheit")
        self.assertAlmostEqual(result, 32.0, places=1)

    def test_celsius_to_fahrenheit_boiling_point(self):
        result = convert_temperature(100, "celsius", "fahrenheit")
        self.assertAlmostEqual(result, 212.0, places=1)

    def test_celsius_to_fahrenheit_negative(self):
        result = convert_temperature(-40, "celsius", "fahrenheit")
        self.assertAlmostEqual(result, -40.0, places=1)

    def test_fahrenheit_to_celsius_freezing_point(self):
        result = convert_temperature(32, "fahrenheit", "celsius")
        self.assertAlmostEqual(result, 0.0, places=1)

    def test_fahrenheit_to_celsius_boiling_point(self):
        result = convert_temperature(212, "fahrenheit", "celsius")
        self.assertAlmostEqual(result, 100.0, places=1)

    def test_fahrenheit_to_celsius_negative(self):
        result = convert_temperature(-40, "fahrenheit", "celsius")
        self.assertAlmostEqual(result, -40.0, places=1)

    def test_celsius_to_kelvin_freezing_point(self):
        result = convert_temperature(0, "celsius", "kelvin")
        self.assertAlmostEqual(result, 273.15, places=1)

    def test_celsius_to_kelvin_boiling_point(self):
        result = convert_temperature(100, "celsius", "kelvin")
        self.assertAlmostEqual(result, 373.15, places=1)

    def test_celsius_to_kelvin_absolute_zero(self):
        result = convert_temperature(-273.15, "celsius", "kelvin")
        self.assertAlmostEqual(result, 0.0, places=1)

    def test_invalid_from_unit(self):
        with self.assertRaises(ValueError):
            convert_temperature(100, "rankine", "fahrenheit")

    def test_invalid_to_unit(self):
        with self.assertRaises(ValueError):
            convert_temperature(100, "celsius", "rankine")

    def test_fahrenheit_to_kelvin_unsupported(self):
        with self.assertRaises(ValueError):
            convert_temperature(32, "fahrenheit", "kelvin")

    def test_kelvin_to_celsius_unsupported(self):
        with self.assertRaises(ValueError):
            convert_temperature(273.15, "kelvin", "celsius")

    def test_invalid_value_type_string(self):
        with self.assertRaises((TypeError, ValueError)):
            convert_temperature("zero", "celsius", "fahrenheit")

    def test_invalid_value_type_none(self):
        with self.assertRaises((TypeError, ValueError)):
            convert_temperature(None, "celsius", "fahrenheit")

    def test_empty_string_from_unit(self):
        with self.assertRaises(ValueError):
            convert_temperature(100, "", "fahrenheit")

    def test_empty_string_to_unit(self):
        with self.assertRaises(ValueError):
            convert_temperature(100, "celsius", "")

    def test_case_sensitivity_from_unit(self):
        with self.assertRaises(ValueError):
            convert_temperature(0, "CELSIUS", "fahrenheit")

    def test_case_sensitivity_to_unit(self):
        with self.assertRaises(ValueError):
            convert_temperature(0, "celsius", "FAHRENHEIT")

    def test_same_unit_conversion(self):
        result = convert_temperature(50, "celsius", "celsius")
        self.assertAlmostEqual(result, 50.0, places=1)

    def test_decimal_precision(self):
        result = convert_temperature(37.0, "celsius", "fahrenheit")
        self.assertAlmostEqual(result, 98.6, places=1)


if __name__ == "__main__":
    unittest.main()
import unittest

from converters import celsius_to_fahrenheit, celsius_to_kelvin, fahrenheit_to_celsius


class TestCelsiusToFahrenheit(unittest.TestCase):
    def test_freezing(self) -> None:
        self.assertAlmostEqual(celsius_to_fahrenheit(0), 32.0)

    def test_boiling(self) -> None:
        self.assertAlmostEqual(celsius_to_fahrenheit(100), 212.0)

    def test_body_temperature(self) -> None:
        self.assertAlmostEqual(celsius_to_fahrenheit(37), 98.6)

    def test_negative(self) -> None:
        self.assertAlmostEqual(celsius_to_fahrenheit(-40), -40.0)


class TestFahrenheitToCelsius(unittest.TestCase):
    def test_freezing(self) -> None:
        self.assertAlmostEqual(fahrenheit_to_celsius(32), 0.0)

    def test_boiling(self) -> None:
        self.assertAlmostEqual(fahrenheit_to_celsius(212), 100.0)

    def test_negative(self) -> None:
        self.assertAlmostEqual(fahrenheit_to_celsius(-40), -40.0)


class TestCelsiusToKelvin(unittest.TestCase):
    def test_freezing(self) -> None:
        self.assertAlmostEqual(celsius_to_kelvin(0), 273.15)

    def test_boiling(self) -> None:
        self.assertAlmostEqual(celsius_to_kelvin(100), 373.15)

    def test_absolute_zero(self) -> None:
        self.assertAlmostEqual(celsius_to_kelvin(-273.15), 0.0)


if __name__ == "__main__":
    unittest.main()

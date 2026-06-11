import pytest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

@pytest.mark.parametrize("celsius, expected", [(0, 32), (100, 212), (-40, -40), (25, 77.0)])
def test_celsius_to_fahrenheit(celsius, expected):
    assert celsius_to_fahrenheit(celsius) == pytest.approx(expected)

@pytest.mark.parametrize("fahrenheit, expected", [(32, 0), (212, 100), (-40, -40), (77, 25)])
def test_fahrenheit_to_celsius(fahrenheit, expected):
    assert fahrenheit_to_celsius(fahrenheit) == pytest.approx(expected)

@pytest.mark.parametrize("celsius, expected", [(0, 273.15), (100, 373.15), (-273.15, 0), (25, 298.15)])
def test_celsius_to_kelvin(celsius, expected):
    assert celsius_to_kelvin(celsius) == pytest.approx(expected, abs=1e-9)
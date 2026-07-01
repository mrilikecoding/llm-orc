import pytest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def test_celsius_to_fahrenheit():
    assert pytest.approx(celsius_to_fahrenheit(0)) == 32
    assert pytest.approx(celsius_to_fahrenheit(100)) == 212
    assert pytest.approx(celsius_to_fahrenheit(-40)) == -40
    assert pytest.approx(celsius_to_fahrenheit(25.5)) == 77.9

def test_fahrenheit_to_celsius():
    assert pytest.approx(fahrenheit_to_celsius(32)) == 0
    assert pytest.approx(fahrenheit_to_celsius(212)) == 100
    assert pytest.approx(fahrenheit_to_celsius(-40)) == -40
    assert pytest.approx(fahrenheit_to_celsius(98.6)) == 37.0

def test_celsius_to_kelvin():
    assert pytest.approx(celsius_to_kelvin(0)) == 273.15
    assert pytest.approx(celsius_to_kelvin(100)) == 373.15
    assert pytest.approx(celsius_to_kelvin(-273.15)) == 0.0
    assert pytest.approx(celsius_to_kelvin(25.5)) == 298.65
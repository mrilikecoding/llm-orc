from converters import (celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin)
import pytest

def test_celsius_to_fahrenheit_0c():
    result = celsius_to_fahrenheit(0)
    assert result == pytest.approx(32)

def test_celsius_to_fahrenheit_100c():
    result = celsius_to_fahrenheit(100)
    assert result == pytest.approx(212)

def test_fahrenheit_to_celsius_32f():
    result = fahrenheit_to_celsius(32)
    assert result == pytest.approx(0)

def test_fahrenheit_to_celsius_212f():
    result = fahrenheit_to_celsius(212)
    assert result == pytest.approx(100)

def test_celsius_to_kelvin_0c():
    result = celsius_to_kelvin(0)
    assert result == pytest.approx(273.15)

def test_celsius_to_kelvin_neg273_15c():
    result = celsius_to_kelvin(-273.15)
    assert result == pytest.approx(0)
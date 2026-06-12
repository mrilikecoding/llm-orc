import pytest
import converters

def test_celsius_to_fahrenheit():
    assert converters.celsius_to_fahrenheit(25) == 77.0

def test_fahrenheit_to_celsius():
    assert converters.fahrenheit_to_celsius(86) == 30.0

def test_celsius_to_kelvin():
    assert converters.celsius_to_kelvin(30) == 303.15
import pytest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(-40) == -40
    assert fahrenheit_to_celsius(celsius_to_fahrenheit(0)) == 0
    assert fahrenheit_to_celsius(celsius_to_fahrenheit(100)) == 100

def test_fahrenheit_to_celsius():
    assert fahrenheit_to_celsius(32) == 0
    assert fahrenheit_to_celsius(212) == 100
    assert fahrenheit_to_celsius(-40) == -40
    assert celsius_to_fahrenheit(fahrenheit_to_celsius(32)) == 32
    assert celsius_to_fahrenheit(fahrenheit_to_celsius(212)) == 212

def test_celsius_to_kelvin():
    assert celsius_to_kelvin(0) == 273.15
    assert celsius_to_kelvin(100) == 373.15
    assert celsius_to_kelvin(-273.15) == 0.0
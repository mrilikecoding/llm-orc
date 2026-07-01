import pytest
import converters

def test_celsius_to_fahrenheit_0():
    assert pytest.approx(converters.celsius_to_fahrenheit(0)) == 32.0

def test_celsius_to_fahrenheit_100():
    assert pytest.approx(converters.celsius_to_fahrenheit(100)) == 212.0

def test_celsius_to_fahrenheit_negative40():
    assert pytest.approx(converters.celsius_to_fahrenheit(-40)) == -40.0

def test_fahrenheit_to_celsius_32():
    assert pytest.approx(converters.fahrenheit_to_celsius(32)) == 0.0

def test_fahrenheit_to_celsius_212():
    assert pytest.approx(converters.fahrenheit_to_celsius(212)) == 100.0

def test_fahrenheit_to_celsius_negative40():
    assert pytest.approx(converters.fahrenheit_to_celsius(-40)) == -40.0

def test_celsius_to_kelvin_0():
    assert pytest.approx(converters.celsius_to_kelvin(0)) == 273.15
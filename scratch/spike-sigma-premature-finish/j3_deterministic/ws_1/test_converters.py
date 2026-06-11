import pytest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(-40) == -40
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37.77777777777777) == pytest.approx(100.0)

def test_fahrenheit_to_celsius():
    assert fahrenheit_to_celsius(-40) == -40
    assert fahrenheit_to_celsius(32) == 0
    assert fahrenheit_to_celsius(212) == 100
    assert fahrenheit_to_celsius(100) == pytest.approx(37.77777777777777)

def test_celsius_to_kelvin():
    assert celsius_to_kelvin(-273.15) == pytest.approx(0.0)
    assert celsius_to_kelvin(0) == pytest.approx(273.15)
    assert celsius_to_kelvin(100) == pytest.approx(373.15)

# Critic's note: The 0K example is not directly testable with the current functions (requires Kelvin-to-Celsius, which is absent). The tests use `pytest.approx` appropriately for floating-point comparisons. No logical errors or API issues are present.
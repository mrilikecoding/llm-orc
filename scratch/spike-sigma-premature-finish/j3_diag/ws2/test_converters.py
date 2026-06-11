import pytest
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == pytest.approx(32)
    assert celsius_to_fahrenheit(100) == pytest.approx(212)
    assert celsius_to_fahrenheit(-40) == pytest.approx(-40)

def test_fahrenheit_to_celsius():
    assert fahrenheit_to_celsius(32) == pytest.approx(0)
    assert fahrenheit_to_celsius(212) == pytest.approx(100)
    assert fahrenheit_to_celsius(-40) == pytest.approx(-40)

def test_celsius_to_kelvin():
    assert celsius_to_kelvin(0) == pytest.approx(273.15)
    assert celsius_to_kelvin(100) == pytest.approx(373.15)
    assert celsius_to_kelvin(-273.15) == pytest.approx(0)

# Critic's Note: The tests cover standard cases and edge cases (e.g., -40°C, absolute zero) using pytest.approx for floating-point comparisons. However, they do not test invalid inputs (e.g., non-numeric values) since the original functions lack input validation. This is acceptable given the problem constraints, but if input validation were required, additional tests would be needed.
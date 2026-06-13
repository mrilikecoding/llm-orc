import converters

def test_celsius_to_fahrenheit():
    assert converters.celsius_to_fahrenheit(0) == 32.0
    assert converters.celsius_to_fahrenheit(100) == 212.0
    assert converters.celsius_to_fahrenheit(-40) == -40.0

def test_fahrenheit_to_celsius():
    assert converters.fahrenheit_to_celsius(32) == 0.0
    assert converters.fahrenheit_to_celsius(212) == 100.0
    assert converters.fahrenheit_to_celsius(98.6) == 37.0

def test_celsius_to_kelvin():
    assert converters.celsius_to_kelvin(0) == 273.15
    assert converters.celsius_to_kelvin(100) == 373.15
    assert converters.celsius_to_kelvin(-273.15) == 0.0
import converters

def test_celsius_to_fahrenheit():
    assert converters.celsius_to_fahrenheit(0) == 32
    assert converters.celsius_to_fahrenheit(100) == 212
    assert converters.celsius_to_fahrenheit(-40) == -40

def test_fahrenheit_to_celsius():
    assert converters.fahrenheit_to_celsius(32) == 0
    assert converters.fahrenheit_to_celsius(212) == 100
    assert converters.fahrenheit_to_celsius(230) == 110
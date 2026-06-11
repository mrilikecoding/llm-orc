def celsius_to_fahrenheit(celsius: float) -> float:
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit: float) -> float:
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius: float) -> float:
    return celsius + 273.15

assert celsius_to_fahrenheit(0) == 32.0
assert celsius_to_fahrenheit(100) == 212.0
assert celsius_to_fahrenheit(-40) == -40.0

assert fahrenheit_to_celsius(32) == 0.0
assert fahrenheit_to_celsius(212) == 100.0
assert fahrenheit_to_celsius(-40) == -40.0

assert celsius_to_kelvin(0) == 273.15
assert celsius_to_kelvin(100) == 373.15
assert celsius_to_kelvin(-273.15) == 0.0
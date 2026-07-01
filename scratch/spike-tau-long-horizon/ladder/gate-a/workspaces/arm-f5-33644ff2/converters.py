def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts Celsius to Fahrenheit using (C * 9/5) + 32"""
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Converts Fahrenheit to Celsius using (F - 32) * 5/9"""
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius: float) -> float:
    """Converts Celsius to Kelvin using C + 273.15"""
    return celsius + 273.15
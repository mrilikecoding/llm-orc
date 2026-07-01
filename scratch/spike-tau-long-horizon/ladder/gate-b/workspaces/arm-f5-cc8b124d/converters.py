def celsius_to_fahrenheit(celsius: Union[int, float]) -> float:
    """Converts Celsius to Fahrenheit using the formula (C * 9/5) + 32."""
    return (celsius * 9/5) + 32


def fahrenheit_to_celsius(fahrenheit: Union[int, float]) -> float:
    """Converts Fahrenheit to Celsius using the formula (F - 32) * 5/9."""
    return (fahrenheit - 32) * 5/9


def celsius_to_kelvin(celsius: Union[int, float]) -> float:
    """Converts Celsius to Kelvin using the formula C + 273.15."""
    return celsius + 273.15
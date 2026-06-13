def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit.
    
    Args:
        celsius: Temperature in degrees Celsius.
        
    Returns:
        Temperature in degrees Fahrenheit.
    """
    return (celsius * 9/5) + 32


def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius.
    
    Args:
        fahrenheit: Temperature in degrees Fahrenheit.
        
    Returns:
        Temperature in degrees Celsius.
    """
    return (fahrenheit - 32) * 5/9


def celsius_to_kelvin(celsius: float) -> float:
    """Convert Celsius to Kelvin.
    
    Args:
        celsius: Temperature in degrees Celsius.
        
    Returns:
        Temperature in Kelvin.
    """
    return celsius + 273.15
def celsius_to_fahrenheit(celsius):
    """
    Convert Celsius to Fahrenheit.
    
    Args:
        celsius (float): Temperature in degrees Celsius.
        
    Returns:
        float: Temperature in degrees Fahrenheit.
    """
    return (celsius * 9/5) + 32

def fahrenheit_to_celsius(fahrenheit):
    """
    Convert Fahrenheit to Celsius.
    
    Args:
        fahrenheit (float): Temperature in degrees Fahrenheit.
        
    Returns:
        float: Temperature in degrees Celsius.
    """
    return (fahrenheit - 32) * 5/9

def celsius_to_kelvin(celsius):
    """
    Convert Celsius to Kelvin.
    
    Args:
        celsius (float): Temperature in degrees Celsius.
        
    Returns:
        float: Temperature in Kelvin.
    """
    return celsius + 273.15
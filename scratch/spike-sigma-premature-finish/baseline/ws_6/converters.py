```python
def celsius_to_fahrenheit(celsius):
    """
    Convert Celsius to Fahrenheit.
    
    Parameters:
    celsius (float): Temperature in Celsius.
    
    Returns:
    float: Temperature in Fahrenheit.
    
    Raises:
    TypeError: If input is not a number.
    """
    if not isinstance(celsius, (int, float)):
        raise TypeError("Input must be a number")
    return (celsius * 9/5) + 32


def fahrenheit_to_celsius(fahrenheit):
    """
    Convert Fahrenheit to Celsius.
    
    Parameters:
    fahrenheit (float): Temperature in Fahrenheit.
    
    Returns:
    float: Temperature in Celsius.
    
    Raises:
    TypeError: If input is not a number.
    """
    if not isinstance(fahrenheit, (int, float)):
        raise TypeError("Input must be a number")
    return (fahrenheit - 32) * 5/9


def celsius_to_kelvin(celsius):
    """
    Convert Celsius to Kelvin.
    
    Parameters:
    celsius (float): Temperature in Celsius.
    
    Returns:
    float: Temperature in Kelvin.
    
    Raises:
    TypeError: If input is not a number.
    """
    if not isinstance(celsius, (int, float)):
        raise TypeError("Input must be a number")
    return celsius + 273.15
```
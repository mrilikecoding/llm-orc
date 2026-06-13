def celsius_to_fahrenheit(temp): 
    """Convert Celsius to Fahrenheit. 
    Formula: (°C × 9/5) + 32 = °F"""
    return (temp * 9/5) + 32

def fahrenheit_to_celsius(temp): 
    """Convert Fahrenheit to Celsius. 
    Formula: (°F - 32) × 5/9 = °C"""
    return (temp - 32) * 5/9

def celsius_to_kelvin(temp): 
    """Convert Celsius to Kelvin. 
    Formula: °C + 273.15 = K"""
    return temp + 273.15

class TestConverters(unittest.TestCase):
    def test_celsius_to_fahrenheit(self): 
        """Test Celsius to Fahrenheit conversion."""
        self.assertAlmostEqual(celsius_to_fahrenheit(0), 32.0)
        self.assertAlmostEqual(celsius_to_fahrenheit(100), 212.0)

    def test_fahrenheit_to_celsius(self): 
        """Test Fahrenheit to Celsius conversion."""
        self.assertAlmostEqual(fahrenheit_to_celsius(32), 0.0)
        self.assertAlmostEqual(fahrenheit_to_celsius(212), 100.0)

    def test_celsius_to_kelvin(self): 
        """Test Celsius to Kelvin conversion."""
        self.assertAlmostEqual(celsius_to_kelvin(0), 273.15)
        self.assertAlmostEqual(celsius_to_kelvin(100), 373.15)

**Critical Issue:** The test module is missing `import unittest` at the top, which is required for the test cases to execute. This will cause NameErrors when running the tests.
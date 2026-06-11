

Temperature Conversion CLI Tool Documentation
==============================================

Installation
------------

1. Ensure Python 3.6 or higher is installed on your system.
2. Download or clone the repository containing the files.
3. Ensure the following files are in the same directory:
   - temperature.py (contains conversion functions)
   - index.js (contains the CLI tool - despite the .js extension, it is a Python file)
4. The tool is ready to use. No additional installation steps are required.

Usage
-----

The temperature-conversion tool converts temperatures between Celsius, Fahrenheit, and Kelvin.

Basic syntax:

    python index.js <value> <from_unit> <to_unit>

Arguments
---------

value         : The temperature value to convert (float or integer).
from_unit     : The source temperature unit (celsius, fahrenheit, kelvin).
to_unit       : The target temperature unit (celsius, fahrenheit, kelvin).

Return Value
------------

The convert() function returns a float representing the converted temperature.

Supported Conversions
---------------------

- celsius to fahrenheit
- fahrenheit to celsius
- celsius to kelvin
- fahrenheit to kelvin (via Celsius conversion)
- kelvin to celsius
- kelvin to fahrenheit (via Celsius conversion)

You can convert from any supported unit to any other supported unit, including converting to the same unit (which returns the original value).

Usage Examples
--------------

Convert 100 Celsius to Fahrenheit:

    python index.js 100 celsius fahrenheit
    Output: 100.0 celsius = 212.0 fahrenheit

Convert 32 Fahrenheit to Celsius:

    python index.js 32 fahrenheit celsius
    Output: 32.0 fahrenheit = 0.0 celsius

Convert 0 Celsius to Kelvin:

    python index.js 0 celsius kelvin
    Output: 0.0 celsius = 273.15 kelvin

Convert 212 Fahrenheit to Celsius:

    python index.js 212 fahrenheit celsius
    Output: 212.0 fahrenheit = 100.0 celsius

Convert -40 Celsius to Fahrenheit (where Celsius and Fahrenheit are equal):

    python index.js -40 celsius fahrenheit
    Output: -40.0 celsius = -40.0 fahrenheit

Convert 300 Kelvin to Celsius:

    python index.js 300 kelvin celsius
    Output: 300.0 kelvin = 26.85 celsius

Convert 300 Kelvin to Fahrenheit:

    python index.js 300 kelvin fahrenheit
    Output: 300.0 kelvin = 80.33 fahrenheit

Convert 100 Fahrenheit to Kelvin:

    python index.js 100 fahrenheit kelvin
    Output: 100.0 fahrenheit = 310.92777777777775 kelvin

Error Handling
--------------

If an unsupported conversion is requested, the tool displays an error message and exits with status code 1.

Example of unsupported conversion:

    python index.js 100 rankine celsius
    Output: Error: Unsupported conversion: rankine to celsius

Using the convert() Function in Python Code
--------------------------------------------

You can import and use the convert function directly in your Python code:

    from index import convert
    
    result = convert(100, 'celsius', 'fahrenheit')
    print(result)  # Output: 212.0

The convert function accepts:
- value: float or int
- from_unit: string ('celsius', 'fahrenheit', or 'kelvin')
- to_unit: string ('celsius', 'fahrenheit', or 'kelvin')

The function raises a ValueError if an unsupported conversion is requested.

Direct Function Access
----------------------

For direct access to specific conversion functions:

    from temperature import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin
    
    fahrenheit = celsius_to_fahrenheit(100)  # Returns 212.0
    celsius = fahrenheit_to_celsius(212)    # Returns 100.0
    kelvin = celsius_to_kelvin(0)           # Returns 273.15

Testing
-------

Run the included tests to verify functionality:

    pytest test_temperature.py    # Tests for conversion functions
    pytest test/cli_test.py       # Tests for CLI tool

The test suite covers:
- Freezing point conversions
- Boiling point conversions
- Absolute zero conversions
- Negative temperatures
- Round-trip conversions (Celsius -> Fahrenheit -> Celsius)
- Edge cases (very high/low temperatures, decimal values)
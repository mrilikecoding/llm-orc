Usage:
To use the temperature-conversion CLI tool, run the following command:

temperature-converter --convert-from <from_unit> --convert-to <to_unit> --value <value>

Where:
<from_unit> is one of celsius, fahrenheit, or kelvin
<to_unit> is one of celsius, fahrenheit, or kelvin
<value> is the temperature value to convert

Examples:
Convert 25 degrees Celsius to Fahrenheit:
temperature-converter --convert-from celsius --convert-to fahrenheit --value 25

Convert 32 degrees Fahrenheit to Celsius:
temperature-converter --convert-from fahrenheit --convert-to celsius --value 32

Convert 0 degrees Celsius to Kelvin:
temperature-converter --convert-from celsius --convert-to kelvin --value 0

Installation:
To install the package, run:
pip install temperature-converter

Alternatively, if you're using the source code, install it via:
pip install -e .

Note:
This tool currently supports the following conversions:
- Celsius to Fahrenheit
- Fahrenheit to Celsius
- Celsius to Kelvin

Additional conversions may be added in future updates.
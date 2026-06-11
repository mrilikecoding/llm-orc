import sys
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

if len(sys.argv) != 3:
    print("Usage: python cli_tool.py <conversion_type> <temperature>")
    sys.exit(1)

conversion_type = sys.argv[1]
try:
    value = float(sys.argv[2])
except ValueError:
    print("Temperature must be a numeric value")
    sys.exit(1)

if conversion_type == "celsius_to_fahrenheit":
    result = celsius_to_fahrenheit(value)
elif conversion_type == "fahrenheit_to_celsius":
    result = fahrenheit_to_celsius(value)
elif conversion_type == "celsius_to_kelvin":
    result = celsius_to_kelvin(value)
else:
    print("Invalid conversion type. Supported types: celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin")
    sys.exit(1)

print(f"Result: {result}")
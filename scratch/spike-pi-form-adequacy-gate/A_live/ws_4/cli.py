```python
import argparse
from converters import *

def main():
    parser = argparse.ArgumentParser(description='Temperature Converter')
    parser.add_argument('--convert-from', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
    parser.add_argument('--convert-to', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
    parser.add_argument('--value', required=True, type=float)
    args = parser.parse_args()

    from_unit = args.convert_from
    to_unit = args.convert_to
    value = args.value

    if from_unit == 'celsius' and to_unit == 'fahrenheit':
        result = celsius_to_fahrenheit(value)
    elif from_unit == 'fahrenheit' and to_unit == 'celsius':
        result = fahrenheit_to_celsius(value)
    elif from_unit == 'celsius' and to_unit == 'kelvin':
        result = celsius_to_kelvin(value)
    else:
        raise ValueError("Unsupported conversion")

    print(result)

if __name__ == '__main__':
    main()
```

**Critical Review:**  
The implementation correctly uses the provided converters but lacks logic for conversions not explicitly handled by the existing functions (e.g., Fahrenheit→Kelvin, Kelvin→Celsius, Kelvin→Fahrenheit). The current code raises `ValueError` for unsupported conversions, which may not be intended. To fully support all unit pairs, additional conversion functions would need to be added to `converters.py`.
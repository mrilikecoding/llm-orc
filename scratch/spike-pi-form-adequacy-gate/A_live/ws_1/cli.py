```python
import argparse
from celsius_converter import *

def main():
    parser = argparse.ArgumentParser(description='Temperature Unit Converter')
    parser.add_argument('--value', type=float, required=True, help='The temperature value to convert')
    parser.add_argument('--from', dest='from_unit', required=True, choices=['celsius', 'fahrenheit', 'kelvin'], help='The source unit')
    parser.add_argument('--to', required=True, choices=['celsius', 'fahrenheit', 'kelvin'], help='The target unit')
    args = parser.parse_args()

    value = args.value
    from_unit = args.from_unit
    to_unit = args.to

    if from_unit == 'celsius' and to_unit == 'fahrenheit':
        result = celsius_to_fahrenheit(value)
    elif from_unit == 'fahrenheit' and to_unit == 'celsius':
        result = fahrenheit_to_celsius(value)
    elif from_unit == 'celsius' and to_unit == 'kelvin':
        result = celsius_to_kelvin(value)
    else:
        print("Unsupported conversion")
        return

    print(f"{value} {from_unit} is equal to {result} {to_unit}")

if __name__ == '__main__':
    main()
```
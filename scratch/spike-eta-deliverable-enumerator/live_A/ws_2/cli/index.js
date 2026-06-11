

import argparse
import sys
from temperature import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def convert(value, from_unit, to_unit):
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    
    if from_unit == to_unit:
        return value
    
    if from_unit == 'celsius' and to_unit == 'fahrenheit':
        return celsius_to_fahrenheit(value)
    
    if from_unit == 'fahrenheit' and to_unit == 'celsius':
        return fahrenheit_to_celsius(value)
    
    if from_unit == 'celsius' and to_unit == 'kelvin':
        return celsius_to_kelvin(value)
    
    if from_unit == 'fahrenheit' and to_unit == 'kelvin':
        celsius = fahrenheit_to_celsius(value)
        return celsius_to_kelvin(celsius)
    
    if from_unit == 'kelvin' and to_unit == 'celsius':
        return value - 273.15
    
    if from_unit == 'kelvin' and to_unit == 'fahrenheit':
        celsius = value - 273.15
        return celsius_to_fahrenheit(celsius)
    
    raise ValueError(f"Unsupported conversion: {from_unit} to {to_unit}")

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units')
    parser.add_argument('value', type=float, help='Temperature value to convert')
    parser.add_argument('from_unit', help='Source unit (celsius, fahrenheit, kelvin)')
    parser.add_argument('to_unit', help='Target unit (celsius, fahrenheit, kelvin)')
    
    args = parser.parse_args()
    
    try:
        result = convert(args.value, args.from_unit, args.to_unit)
        print(f"{args.value} {args.from_unit} = {result} {args.to_unit}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
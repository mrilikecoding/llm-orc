

#!/usr/bin/env python3
import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

VALID_UNITS = ['celsius', 'fahrenheit', 'kelvin']

def convert(value, from_unit, to_unit):
    if from_unit == to_unit:
        return value
    
    if from_unit == 'celsius':
        if to_unit == 'fahrenheit':
            return celsius_to_fahrenheit(value)
        elif to_unit == 'kelvin':
            return celsius_to_kelvin(value)
    elif from_unit == 'fahrenheit':
        if to_unit == 'celsius':
            return fahrenheit_to_celsius(value)
        elif to_unit == 'kelvin':
            celsius = fahrenheit_to_celsius(value)
            return celsius_to_kelvin(celsius)
    elif from_unit == 'kelvin':
        if to_unit == 'celsius':
            return value - 273.15
        elif to_unit == 'fahrenheit':
            celsius = value - 273.15
            return celsius_to_fahrenheit(celsius)
    
    raise ValueError(f"Cannot convert from {from_unit} to {to_unit}")

def main():
    parser = argparse.ArgumentParser(description='Temperature unit converter')
    parser.add_argument('--from', dest='from_unit', required=True, help='Source unit')
    parser.add_argument('--to', dest='to_unit', required=True, help='Target unit')
    parser.add_argument('--value', type=float, required=True, help='Value to convert')
    
    args = parser.parse_args()
    
    from_unit = args.from_unit.lower()
    to_unit = args.to_unit.lower()
    
    if from_unit not in VALID_UNITS:
        print(f"Error: Invalid source unit '{args.from_unit}'. Valid units: {', '.join(VALID_UNITS)}")
        return 1
    
    if to_unit not in VALID_UNITS:
        print(f"Error: Invalid target unit '{args.to_unit}'. Valid units: {', '.join(VALID_UNITS)}")
        return 1
    
    result = convert(args.value, from_unit, to_unit)
    print(f"{args.value} {from_unit} = {result} {to_unit}")
    
    return 0

if __name__ == '__main__':
    exit(main())
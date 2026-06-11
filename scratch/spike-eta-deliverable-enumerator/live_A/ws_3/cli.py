

import argparse
import sys
sys.path.insert(0, '/Users/nathangreen/Development/eddi-lab/llm-orc/scratch/spike-eta-deliverable-enumerator/live_A/ws_3')
from temp_convert import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin


def kelvin_to_celsius(kelvin: float) -> float:
    return kelvin - 273.15


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    
    if from_unit == to_unit:
        return value
    
    # All conversions go through Celsius
    if from_unit == 'C':
        celsius = value
    elif from_unit == 'F':
        celsius = fahrenheit_to_celsius(value)
    elif from_unit == 'K':
        celsius = kelvin_to_celsius(value)
    else:
        raise ValueError(f"Unknown source unit: {from_unit}")
    
    # Convert from Celsius to target
    if to_unit == 'C':
        return celsius
    elif to_unit == 'F':
        return celsius_to_fahrenheit(celsius)
    elif to_unit == 'K':
        return celsius_to_kelvin(celsius)
    else:
        raise ValueError(f"Unknown target unit: {to_unit}")


def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius, Fahrenheit, and Kelvin')
    parser.add_argument('value', type=float, help='Temperature value to convert')
    parser.add_argument('from_unit', help='Source unit (C, F, or K)')
    parser.add_argument('to_unit', help='Target unit (C, F, or K)')
    
    args = parser.parse_args()
    
    result = convert_temperature(args.value, args.from_unit, args.to_unit)
    print(f"{args.value}{args.from_unit.upper()} = {result}{args.to_unit.upper()}")


if __name__ == '__main__':
    main()
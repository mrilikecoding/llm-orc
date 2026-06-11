

#!/usr/bin/env python3
"""Command-line tool for temperature conversions."""
import argparse
import sys
from temperature import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin


# Available conversions using only the module's functions:
# - C -> F (direct)
# - F -> C (direct)
# - C -> K (direct)
# - F -> K (via C: fahrenheit_to_celsius -> celsius_to_kelvin)
#
# Not possible without kelvin_to_celsius:
# - K -> C, K -> F


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between different units using available module functions."""
    from_u = from_unit.upper()
    to_u = to_unit.upper()

    # Same unit - no conversion needed
    if from_u == to_u:
        return value

    # C -> F
    if from_u == 'C' and to_u == 'F':
        return celsius_to_fahrenheit(value)

    # F -> C
    if from_u == 'F' and to_u == 'C':
        return fahrenheit_to_celsius(value)

    # C -> K
    if from_u == 'C' and to_u == 'K':
        return celsius_to_kelvin(value)

    # F -> K (via Celsius using available functions)
    if from_u == 'F' and to_u == 'K':
        celsius = fahrenheit_to_celsius(value)
        return celsius_to_kelvin(celsius)

    # K -> C and K -> F not possible without inverse Kelvin function
    raise ValueError(
        f"Conversion from {from_unit} to {to_unit} not supported. "
        f"Available: C<->F, C<->K, F->K"
    )


def main():
    parser = argparse.ArgumentParser(
        description='Convert temperature between Celsius, Fahrenheit, and Kelvin.'
    )
    parser.add_argument(
        'temperature',
        type=float,
        help='Temperature value to convert'
    )
    parser.add_argument(
        '-i', '--input-unit',
        choices=['C', 'F', 'K'],
        default='C',
        help='Input temperature unit (default: C)'
    )
    parser.add_argument(
        '-o', '--output-unit',
        choices=['C', 'F', 'K'],
        default='F',
        help='Output temperature unit (default: F)'
    )

    args = parser.parse_args()

    try:
        result = convert_temperature(args.temperature, args.input_unit, args.output_unit)
        print(f"{args.temperature}{args.input_unit} = {result}{args.output_unit}")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
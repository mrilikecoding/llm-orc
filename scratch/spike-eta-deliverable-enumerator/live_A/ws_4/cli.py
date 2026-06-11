

#!/usr/bin/env python3
"""Command-line tool for temperature conversion."""

import argparse
from temperature_conversion import (
    celsius_to_fahrenheit,
    fahrenheit_to_celsius,
    celsius_to_kelvin,
)


def main():
    parser = argparse.ArgumentParser(description="Temperature conversion tool")
    parser.add_argument(
        "--convert-from",
        choices=["celsius", "fahrenheit"],
        required=True,
        help="Temperature unit to convert from",
    )
    parser.add_argument(
        "--convert-to",
        choices=["fahrenheit", "celsius", "kelvin"],
        required=True,
        help="Temperature unit to convert to",
    )
    parser.add_argument(
        "value",
        type=float,
        help="Temperature value to convert",
    )

    args = parser.parse_args()

    if args.convert_from == args.convert_to:
        result = args.value
    elif args.convert_from == "celsius":
        if args.convert_to == "fahrenheit":
            result = celsius_to_fahrenheit(args.value)
        elif args.convert_to == "kelvin":
            result = celsius_to_kelvin(args.value)
    elif args.convert_from == "fahrenheit":
        if args.convert_to == "celsius":
            result = fahrenheit_to_celsius(args.value)
        elif args.convert_to == "kelvin":
            celsius = fahrenheit_to_celsius(args.value)
            result = celsius_to_kelvin(celsius)

    print(f"{args.value} {args.convert_from} = {result} {args.convert_to}")


if __name__ == "__main__":
    main()
import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius, Fahrenheit, and Kelvin.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('from_unit', choices=['C', 'F', 'K'], help='The unit of the input temperature.')
    parser.add_argument('to_unit', choices=['C', 'F', 'K'], help='The unit to convert to.')
    args = parser.parse_args()

    allowed_conversions = {
        ('C', 'F'): converters.celsius_to_fahrenheit,
        ('F', 'C'): converters.fahrenheit_to_celsius,
        ('C', 'K'): converters.celsius_to_kelvin,
    }

    if (args.from_unit, args.to_unit) not in allowed_conversions:
        parser.error(f"Conversion from {args.from_unit} to {args.to_unit} is not supported.")

    conversion_func = allowed_conversions[(args.from_unit, args.to_unit)]
    result = conversion_func(args.value)

    print(f"{args.value} {args.from_unit} is equal to {result} {args.to_unit}.")

if __name__ == '__main__':
    main()

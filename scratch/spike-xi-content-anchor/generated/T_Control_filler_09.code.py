import argparse
import converters

def main() -> None:
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius and Fahrenheit.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('from_unit', choices=['celsius', 'fahrenheit'], help='The unit to convert from.')
    parser.add_argument('to_unit', choices=['celsius', 'fahrenheit'], help='The unit to convert to.')
    args = parser.parse_args()

    if args.from_unit == args.to_unit:
        parser.error("Cannot convert from and to the same unit.")

    if args.from_unit == 'celsius' and args.to_unit == 'fahrenheit':
        result = converters.celsius_to_fahrenheit(args.value)
    elif args.from_unit == 'fahrenheit' and args.to_unit == 'celsius':
        result = converters.fahrenheit_to_celsius(args.value)
    else:
        parser.error(f"Unsupported conversion from {args.from_unit} to {args.to_unit}.")

    print(result)

if __name__ == '__main__':
    main()

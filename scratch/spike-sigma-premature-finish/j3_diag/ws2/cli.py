import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert temperatures between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('--from', '-f', required=True, choices=['celsius', 'fahrenheit'], help='The unit to convert from.')
    parser.add_argument('--to', '-t', required=True, choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert to.')
    args = parser.parse_args()

    if args.from == 'celsius' and args.to == 'fahrenheit':
        result = celsius_to_fahrenheit(args.value)
    elif args.from == 'celsius' and args.to == 'kelvin':
        result = celsius_to_kelvin(args.value)
    elif args.from == 'fahrenheit' and args.to == 'celsius':
        result = fahrenheit_to_celsius(args.value)
    else:
        raise ValueError(f"Conversion from {args.from} to {args.to} is not supported.")

    print(f"{args.value} {args.from} is equal to {result} {args.to}")

if __name__ == '__main__':
    main()
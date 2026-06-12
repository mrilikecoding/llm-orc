import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature values.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('source', choices=['celsius', 'fahrenheit'], help='The source unit.')
    parser.add_argument('target', choices=['fahrenheit', 'celsius', 'kelvin'], help='The target unit.')
    args = parser.parse_args()

    if args.source == 'celsius':
        if args.target == 'fahrenheit':
            result = converters.celsius_to_fahrenheit(args.value)
        elif args.target == 'kelvin':
            result = converters.celsius_to_kelvin(args.value)
        else:
            parser.error(f"Invalid target unit: {args.target}. Supported: fahrenheit, kelvin")
    elif args.source == 'fahrenheit':
        if args.target == 'celsius':
            result = converters.fahrenheit_to_celsius(args.value)
        else:
            parser.error(f"Invalid target unit: {args.target}. Supported: celsius")
    else:
        parser.error(f"Invalid source unit: {args.source}. Supported: celsius, fahrenheit")

    print(f"{args.value} {args.source} is equal to {result} {args.target}")

if __name__ == '__main__':
    main()
import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert temperature values.')
    parser.add_argument('--from', dest='from_scale', required=True, choices=['celsius', 'fahrenheit', 'kelvin'], help='Source temperature scale')
    parser.add_argument('--to', dest='to_scale', required=True, choices=['celsius', 'fahrenheit', 'kelvin'], help='Target temperature scale')
    parser.add_argument('--value', type=float, required=True, help='Temperature value to convert')
    args = parser.parse_args()

    if args.from_scale == 'celsius' and args.to_scale == 'fahrenheit':
        result = celsius_to_fahrenheit(args.value)
    elif args.from_scale == 'fahrenheit' and args.to_scale == 'celsius':
        result = fahrenheit_to_celsius(args.value)
    elif args.from_scale == 'celsius' and args.to_scale == 'kelvin':
        result = celsius_to_kelvin(args.value)
    else:
        print(f"Error: Unsupported conversion from {args.from_scale} to {args.to_scale}")
        return

    print(f"Converted value: {result:.2f}")

if __name__ == '__main__':
    main()
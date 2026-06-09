import argparse
import converters

def main(args: argparse.Namespace) -> None:
    if args.from_unit == args.to_unit:
        print(args.value)
        return

    if args.from_unit == 'C' and args.to_unit == 'F':
        result = converters.celsius_to_fahrenheit(args.value)
    elif args.from_unit == 'C' and args.to_unit == 'K':
        result = converters.celsius_to_kelvin(args.value)
    elif args.from_unit == 'F' and args.to_unit == 'C':
        result = converters.fahrenheit_to_celsius(args.value)
    elif args.from_unit == 'F' and args.to_unit == 'K':
        result = converters.fahrenheit_to_kelvin(args.value)
    elif args.from_unit == 'K' and args.to_unit == 'C':
        result = converters.kelvin_to_celsius(args.value)
    elif args.from_unit == 'K' and args.to_unit == 'F':
        result = converters.kelvin_to_fahrenheit(args.value)
    else:
        raise ValueError("Unsupported unit conversion")

    print(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('--from', dest='from_unit', required=True, choices=['C', 'F', 'K'], help='The input unit (C, F, K).')
    parser.add_argument('--to', dest='to_unit', required=True, choices=['C', 'F', 'K'], help='The target unit (C, F, K).')
    args = parser.parse_args()
    main(args)

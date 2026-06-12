import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Temperature conversion tool')
    subparsers = parser.add_subparsers(dest='command')

    # Celsius to Fahrenheit
    c2f_parser = subparsers.add_parser('c2f', help='Convert Celsius to Fahrenheit')
    c2f_parser.add_argument('celsius', type=float, help='Temperature in Celsius')

    # Fahrenheit to Celsius
    f2c_parser = subparsers.add_parser('f2c', help='Convert Fahrenheit to Celsius')
    f2c_parser.add_argument('fahrenheit', type=float, help='Temperature in Fahrenheit')

    # Celsius to Kelvin
    c2k_parser = subparsers,add_parser('c2k', help='Convert Celsius to Kelvin')
    c2k_parser.add_argument('celsius', type=float, help='Temperature in Celsius')

    args = parser.parse_args()

    if args.command == 'c2f':
        result = celsius_to_fahrenheit(args.celsius)
    elif args.command == 'f2c':
        result = fahrenheit_to_celsius(args.fahrenheit)
    elif args.command == 'c2k':
        result = celsius_to_kelvin(args.celsius)

    print(f'Result: {result}')

if __name__ == '__main__':
    main()
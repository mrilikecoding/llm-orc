import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert temperatures between units')
    subparsers = parser.add_subparsers(dest='command')

    # Celsius to Fahrenheit
    c_to_f_parser = subparsers.add_parser('celsius-to-fahrenheit')
    c_to_f_parser.add_argument('value', type=float, help='Temperature in Celsius')
    c_to_f_parser.set_defaults(func=celsius_to_fahrenheit)

    # Fahrenheit to Celsius
    f_to_c_parser = subparsers.add_parser('fahrenheit-to-celsius')
    f_to_c_parser.add_argument('value', type=float, help='Temperature in Fahrenheit')
    f_to_c_parser.set_defaults(func=fahrenheit_to_celsius)

    # Celsius to Kelvin
    c_to_k_parser = subparsers.add_parser('celsius-to-kelvin')
    c_to_k_parser.add_argument('value', type=float, help='Temperature in Celsius')
    c_to_k_parser.set_defaults(func=celsius_to_kelvin)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        result = args.func(args.value)
        print(result)
    else:
        parser.error('Invalid command')

if __name__ == '__main__':
    main()
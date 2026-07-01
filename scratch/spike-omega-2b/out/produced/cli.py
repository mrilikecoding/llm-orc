import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Temperature conversion CLI')
    subparsers = parser.add_subparsers(dest='command')

    # c-to-f
    parser_c_to_f = subparsers.add_parser('c-to-f', help='Convert Celsius to Fahrenheit')
    parser_c_to_f.add_argument('celsius', type=float, help='Temperature in Celsius')
    parser_c_to_f.set_defaults(func=celsius_to_fahrenheit)

    # f-to-c
    parser_f_to_c = subparsers.add_parser('f-to-c', help='Convert Fahrenheit to Celsius')
    parser_f_to_c.add_argument('fahrenheit', type=float, help='Temperature in Fahrenheit')
    parser_f_to_c.set_defaults(func=fahrenheit_to_celsius)

    # c-to-k
    parser_c_to_k = subparsers.add_parser('c-to-k', help='Convert Celsius to Kelvin')
    parser_c_to_k.add_argument('celsius', type=float, help='Temperature in Celsius')
    parser_c_to_k.set_defaults(func=celsius_to_kelvin)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        result = args.func(args.celsius) if hasattr(args, 'celsius') else args.func(args.fahrenheit)
        print(result)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
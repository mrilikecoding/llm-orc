def celsius_to_fahrenheit(c):
    return (c * 9/5) + 32

def fahrenheit_to_celsius(f):
    return (f - 32) * 5/9

def celsius_to_kelvin(c):
    return c + 273.15

import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Temperature conversion tools')
    subparsers = parser.add_subparsers(dest='command')

    parser_c_to_f = subparsers.add_parser('c-to-f', help='Convert Celsius to Fahrenheit')
    parser_c_to_f.add_argument('value', type=float, help='Temperature in Celsius')

    parser_f_to_c = subparsers.add_parser('f-to-c', help='Convert Fahrenheit to Celsius')
    parser_f_to_c.add_argument('value', type=float, help='Temperature in Fahrenheit')

    parser_c_to_k = subparsers.add_parser('c-to-k', help='Convert Celsius to Kelvin')
    parser_c_to_k.add_argument('value', type=float, help='Temperature in Celsius')

    args = parser.parse_args()
    if args.command == 'c-to-f':
        result = converters.celsius_to_fahrenheit(args.value)
    elif args.command == 'f-to-c':
        result = converters.fahrenheit_to_celsius(args.value)
    elif args.command == 'c-to-k':
        result = converters.celsius_to_kelvin(args.value)
    print(result)

if __name__ == '__main__':
    main()
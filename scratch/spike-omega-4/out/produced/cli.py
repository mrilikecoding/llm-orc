import argparse
from converters import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--celsius-to-fahrenheit', type=float)
    parser.add_argument('--fahrenheit-to-celsius', type=float)
    parser.add_argument('--celsius-to-kelvin', type=float)

    args = parser.parse_args()

    if args.celsius_to_fahrenheit is not None:
        print(celsius_to_fahrenheit(args.celsius_to_fahrenheit))
    elif args.fahrenheit_to_celsius is not None:
        print(fahrenheit_to_celsius(args.fahrenheit_to_celsius))
    elif args.celsius_to_kelvin is not None:
        print(celsius_to_kelvin(args.celsius_to_kelvin))
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
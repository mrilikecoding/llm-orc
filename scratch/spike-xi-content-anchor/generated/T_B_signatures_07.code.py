import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units')
    parser.add_argument('value', type=float, help='The temperature value to convert')
    parser.add_argument('target_unit', choices=['F', 'C', 'K'], help='Target unit: F, C, or K')
    args = parser.parse_args()

    if args.target_unit == 'F':
        result = converters.celsius_to_fahrenheit(args.value)
    elif args.target_unit == 'C':
        result = converters.fahrenheit_to_celsius(args.value)
    elif args.target_unit == 'K':
        result = converters.celsius_to_kelvin(args.value)
    print(result)

if __name__ == '__main__':
    main()

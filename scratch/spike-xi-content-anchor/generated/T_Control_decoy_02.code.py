import argparse
from converters import convert_temperature

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('--from-unit', required=True, help='The original unit (e.g., Celsius, Fahrenheit, Kelvin)')
    parser.add_argument('--to-unit', required=True, help='The target unit (e.g., Celsius, Fahrenheit, Kelvin)')
    args = parser.parse_args()
    converted = convert_temperature(args.value, args.from_unit, args.to_unit)
    print(converted)

if __name__ == '__main__':
    main()

import argparse
import converters

def main() -> None:
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('--value', type=float, required=True, help='The temperature value to convert.')
    parser.add_argument('--from', dest='from_unit', required=True, choices=['c', 'f'], help='The unit to convert from (c for Celsius, f for Fahrenheit).')
    parser.add_argument('--to', required=True, choices=['c', 'f'], help='The unit to convert to (c for Celsius, f for Fahrenheit).')
    args = parser.parse_args()

    result = converters.convert_temperature(args.value, args.from_unit, args.to)
    print(result)

if __name__ == '__main__':
    main()

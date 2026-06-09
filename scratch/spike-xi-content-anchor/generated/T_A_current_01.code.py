import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('from_unit', choices=['C', 'F'], help='The unit to convert from (C or F).')
    parser.add_argument('to_unit', choices=['C', 'F'], help='The unit to convert to (C or F).')
    args = parser.parse_args()

    result = converters.convert(args.value, args.from_unit, args.to_unit)
    print(result)

if __name__ == '__main__':
    main()

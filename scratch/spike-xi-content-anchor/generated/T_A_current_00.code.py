import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius and Fahrenheit')
    parser.add_argument('value', type=float, help='Temperature value')
    parser.add_argument('from_unit', choices=['C', 'F'], help='Input unit (Celsius or Fahrenheit)')
    parser.add_argument('to_unit', choices=['C', 'F'], help='Output unit')
    args = parser.parse_args()

    if args.from_unit == 'C' and args.to_unit == 'F':
        result = converters.celsius_to_fahrenheit(args.value)
    elif args.from_unit == 'F' and args.to_unit == 'C':
        result = converters.fahrenheit_to_celsius(args.value)
    else:
        print("Unsupported conversion")
        return

    print(f"{args.value} {args.from_unit} is equal to {result} {args.to_unit}")

if __name__ == '__main__':
    main()

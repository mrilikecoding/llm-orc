import argparse

from converters import celsius_to_fahrenheit, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert Celsius to Fahrenheit or Kelvin')
    parser.add_argument('value', type=float, help='The Celsius value to convert')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-fahrenheit', action='store_true', help='Convert to Fahrenheit')
    group.add_argument('--to-kelvin', action='store_true', help='Convert to Kelvin')
    args = parser.parse_args()
    value = args.value
    if args.to_fahrenheit:
        converted = celsius_to_fahrenheit(value)
    else:
        converted = celsius_to_kelvin(value)
    print(converted)

if __name__ == '__main__':
    main()
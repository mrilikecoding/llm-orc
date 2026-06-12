import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert temperature values.')
    parser.add_argument('--value', type=float, required=True, help='The temperature value to convert.')
    parser.add_argument('--from', dest='from_unit', choices=['c', 'f', 'k'], required=True, help='The unit to convert from.')
    parser.add_argument('--to', choices=['c', 'f', 'k'], required=True, help='The unit to convert to.')
    args = parser.parse_args()

    value = args.value
    from_unit = args.from_unit
    to_unit = args.to

    if from_unit == 'c' and to_unit == 'f':
        result = celsius_to_fahrenheit(value)
    elif from_unit == 'f' and to_unit == 'c':
        result = fahrenheit_to_celsius(value)
    elif from_unit == 'c' and to_unit == 'k':
        result = celsius_to_kelvin(value)
    else:
        print("Unsupported conversion.")
        return

    print(result)

if __name__ == '__main__':
    main()
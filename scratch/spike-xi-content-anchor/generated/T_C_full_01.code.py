import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert temperature.')
    parser.add_argument('--value', type=float, required=True, help='The input value.')
    parser.add_argument('--to', choices=['fahrenheit', 'celsius', 'kelvin'], required=True, help='Target unit.')
    args = parser.parse_args()

    value = args.value
    target = args.to

    if target == 'fahrenheit':
        result = celsius_to_fahrenheit(value)
    elif target == 'celsius':
        result = fahrenheit_to_celsius(value)
    elif target == 'kelvin':
        result = celsius_to_kelvin(value)

    print(result)

if __name__ == '__main__':
    main()

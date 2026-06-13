import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Temperature Conversion CLI')
    parser.add_argument('value', type=float, help='The temperature value to convert')
    parser.add_argument('source_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit of the input temperature')
    parser.add_argument('target_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert to')
    args = parser.parse_args()

    value = args.value
    source = args.source_unit
    target = args.target_unit

    if source == target:
        print(f"{value} {source} is equal to {value} {target}")
        return

    if source == 'celsius' and target == 'fahrenheit':
        result = celsius_to_fahrenheit(value)
    elif source == 'fahrenheit' and target == 'celsius':
        result = fahrenheit_to_celsius(value)
    elif source == 'celsius' and target == 'kelvin':
        result = celsius_to_kelvin(value)
    elif source == 'kelvin' and target == 'celsius':
        result = value - 273.15
    elif source == 'fahrenheit' and target == 'kelvin':
        celsius = fahrenheit_to_celsius(value)
        result = celsius_to_kelvin(celsius)
    elif source == 'kelvin' and target == 'fahrenheit':
        celsius = value - 273.15
        result = celsius_to_fahrenheit(celsius)
    else:
        print("Unsupported conversion")
        return

    print(f"{value} {source} is equal to {result} {target}")

if __name__ == '__main__':
    main()
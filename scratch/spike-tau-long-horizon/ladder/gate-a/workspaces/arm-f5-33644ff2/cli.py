import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('from_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert from.')
    parser.add_argument('to_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert to.')
    args = parser.parse_args()

    value = args.value
    from_unit = args.from_unit
    to_unit = args.to_unit

    if from_unit == 'celsius':
        if to_unit == 'fahrenheit':
            result = celsius_to_fahrenheit(value)
        elif to_unit == 'kelvin':
            result = celsius_to_kelvin(value)
        else:
            result = value
    elif from_unit == 'fahrenheit':
        if to_unit == 'celsius':
            result = fahrenheit_to_celsius(value)
        elif to_unit == 'kelvin':
            celsius = fahrenheit_to_celsius(value)
            result = celsius_to_kelvin(celsius)
        else:
            result = value
    elif from_unit == 'kelvin':
        if to_unit == 'celsius':
            result = value - 273.15
        elif to_unit == 'fahrenheit':
            celsius = value - 273.15
            result = celsius_to_fahrenheit(celsius)
        else:
            result = value
    else:
        parser.error("Invalid from_unit")

    print(result)

if __name__ == '__main__':
    main()
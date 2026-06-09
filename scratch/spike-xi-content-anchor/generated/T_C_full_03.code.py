import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('from_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert from.')
    parser.add_argument('to_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert to.')
    args = parser.parse_args()

    value = args.value
    from_unit = args.from_unit
    to_unit = args.to_unit

    allowed = {
        ('celsius', 'fahrenheit'): converters.celsius_to_fahrenheit,
        ('celsius', 'kelvin'): converters.celsius_to_kelvin,
        ('fahrenheit', 'celsius'): converters.fahrenheit_to_celsius,
    }

    if (from_unit, to_unit) in allowed:
        result = allowed[(from_unit, to_unit)](value)
    else:
        print("Invalid conversion. Supported conversions: celsius to fahrenheit, celsius to kelvin, fahrenheit to celsius.")
        return

    print(result)

if __name__ == '__main__':
    main()

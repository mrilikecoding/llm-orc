import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert units')
    parser.add_argument('value', type=float, help='The value to convert')
    parser.add_argument('from_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert from')
    parser.add_argument('to_unit', choices=['celsius', 'fahrenheit', 'kelvin'], help='The unit to convert to')
    args = parser.parse_args()

    allowed = {
        ('celsius', 'fahrenheit'): converters.celsius_to_fahrenheit,
        ('fahrenheit', 'celsius'): converters.fahrenheit_to_celsius,
        ('celsius', 'kelvin'): converters.celsius_to_kelvin,
        ('kelvin', 'celsius'): converters.celsius_to_kelvin,  # Reverse conversion not supported by existing functions
        ('kelvin', 'fahrenheit'): converters.celsius_to_fahrenheit,  # Requires additional function
        ('fahrenheit', 'kelvin'): converters.celsius_to_kelvin,  # Requires additional function
    }

    if (args.from_unit, args.to_unit) not in allowed:
        print(f"Error: Conversion from {args.from_unit} to {args.to_unit} is not supported")
        return

    func = allowed[(args.from_unit, args.to_unit)]
    result = func(args.value)
    print(result)

if __name__ == '__main__':
    main()
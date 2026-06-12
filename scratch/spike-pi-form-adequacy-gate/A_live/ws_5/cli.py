```python
import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--from-celsius', action='store_true')
    group.add_argument('--from-fahrenheit', action='store_true')
    group.add_argument('--from-kelvin', action='store_true')
    parser.add_argument('--to', required=True, choices=['celsius', 'fahrenheit', 'kelvin'])
    parser.add_argument('temperature', type=float, help='Temperature value')

    args = parser.parse_args()

    from_unit = None
    if args.from_celsius:
        from_unit = 'celsius'
    elif args.from_fahrenheit:
        from_unit = 'fahrenheit'
    elif args.from_kelvin:
        from_unit = 'kelvin'

    to_unit = args.to
    temp = args.temperature

    if from_unit == 'celsius' and to_unit == 'fahrenheit':
        result = converters.celsius_to_fahrenheit(temp)
    elif from_unit == 'fahrenheit' and to_unit == 'celsius':
        result = converters.fahrenheit_to_celsius(temp)
    elif from_unit == 'celsius' and to_unit == 'kelvin':
        result = converters.celsius_to_kelvin(temp)
    elif from_unit == 'fahrenheit' and to_unit == 'kelvin':
        result = converters.celsius_to_kelvin(converters.fahrenheit_to_celsius(temp))
    elif from_unit == 'kelvin' and to_unit == 'celsius':
        result = temp - 273.15
    elif from_unit == 'kelvin' and to_unit == 'fahrenheit':
        result = converters.celsius_to_fahrenheit(temp - 273.15)
    else:
        parser.error(f"Unsupported conversion from {from_unit} to {to_unit}")

    print(f"{temp} {from_unit} is {result} {to_unit}")

if __name__ == '__main__':
    main()
```
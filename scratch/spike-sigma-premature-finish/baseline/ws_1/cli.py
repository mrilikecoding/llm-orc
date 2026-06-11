```python
import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Temperature conversion CLI.')
    parser.add_argument('value', type=float, help='Temperature value to convert.')
    parser.add_argument('conversion', choices=['celsius_to_fahrenheit', 'fahrenheit_to_celsius', 'celsius_to_kelvin'], help='Conversion type.')
    args = parser.parse_args()

    if args.conversion == 'celsius_to_fahrenheit':
        result = converters.celsius_to_fahrenheit(args.value)
    elif args.conversion == 'fahrenheit_to_celsius':
        result = converters.fahrenheit_to_celsius(args.value)
    elif args.conversion == 'celsius_to_kelvin':
        result = converters.celsius_to_kelvin(args.value)
    else:
        parser.error("Invalid conversion type")

    print(f"Result: {result}")

if __name__ == '__main__':
    main()
```
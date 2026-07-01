import argparse
from converters import celsius_to_fahrenheit, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert Celsius to Fahrenheit or Kelvin')
    parser.add_argument('celsius', type=float, help='Temperature in Celsius')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-fahrenheit', action='store_true', help='Convert to Fahrenheit')
    group.add_argument('--to-kelvin', action='store_true', help='Convert to Kelvin')
    
    args = parser.parse_args()
    
    if args.to_fahrenheit:
        result = celsius_to_fahrenheit(args.celsius)
    else:
        result = celsius_to_kelvin(args.celsius)
    
    print(f'Result: {result}')

if __name__ == '__main__':
    main()
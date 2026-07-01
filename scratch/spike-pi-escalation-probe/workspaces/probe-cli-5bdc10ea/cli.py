"""Convert temperature between Celsius and Fahrenheit using command line arguments."""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius and Fahrenheit.')
    parser.add_argument('temperature', type=float, help='Temperature value')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-fahrenheit', action='store_true', help='Convert to Fahrenheit')
    group.add_argument('--to-celsius', action='store_true', help='Convert to Celsius')
    args = parser.parse_args()

    temp = args.temperature
    if args.to_fahrenheit:
        result = (temp * 9/5) + 32
    else:
        result = (temp - 32) * 5/9

    print(f"{result:.2f}")

if __name__ == '__main__':
    main()
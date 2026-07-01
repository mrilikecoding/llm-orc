"""Convert temperature between Celsius and Fahrenheit using command-line arguments."""

import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius and Fahrenheit.')
    parser.add_argument('temperature', type=float, help='Temperature value')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-fahrenheit', action='store_true', help='Convert to Fahrenheit')
    group.add_argument('--to-celsius', action='store_true', help='Convert to Celsius')
    args = parser.parse_args()

    if args.to_fahrenheit:
        f = (args.temperature * 9/5) + 32
        print(f"{f:.2f}")
    else:  # args.to_celsius is True
        c = (args.temperature - 32) * 5/9
        print(f"{c:.2f}")

if __name__ == '__main__':
    main()
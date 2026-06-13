import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius and Fahrenheit.')
    parser.add_argument('temp', type=float, help='Temperature value')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--to-fahrenheit', action='store_true', help='Convert to Fahrenheit')
    group.add_argument('--to-celsius', action='store_true', help='Convert to Celsius')
    args = parser.parse_args()

    if args.to_fahrenheit:
        # Convert Celsius to Fahrenheit
        result = (args.temp * 9/5) + 32
    else:
        # Convert Fahrenheit to Celsius
        result = (args.temp - 32) * 5/9

    print(f"{result:.2f}")

if __name__ == "__main__":
    main()
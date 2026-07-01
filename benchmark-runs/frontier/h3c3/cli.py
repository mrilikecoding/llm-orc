import argparse

from converters import celsius_to_fahrenheit, celsius_to_kelvin, fahrenheit_to_celsius


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert temperatures between units."
    )
    parser.add_argument("value", type=float, help="Temperature value to convert")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--to-fahrenheit",
        action="store_true",
        help="Convert Celsius to Fahrenheit",
    )
    group.add_argument(
        "--to-kelvin",
        action="store_true",
        help="Convert Celsius to Kelvin",
    )
    group.add_argument(
        "--to-celsius",
        action="store_true",
        help="Convert Fahrenheit to Celsius",
    )

    args = parser.parse_args()

    if args.to_fahrenheit:
        result = celsius_to_fahrenheit(args.value)
        print(result)
    elif args.to_kelvin:
        result = celsius_to_kelvin(args.value)
        print(result)
    elif args.to_celsius:
        result = fahrenheit_to_celsius(args.value)
        print(result)


if __name__ == "__main__":
    main()

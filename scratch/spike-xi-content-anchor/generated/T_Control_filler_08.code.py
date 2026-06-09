import argparse
import converters

def main() -> None:
    """Main entry point for the temperature conversion CLI."""
    parser = argparse.ArgumentParser(description='Convert temperature values between units.')
    parser.add_argument('--value', type=float, required=True, help='The temperature value to convert.')
    parser.add_argument('--to', type=str, required=True, choices=['c', 'f', 'k'], 
                        help='Target unit: c (Celsius), f (Fahrenheit), k (Kelvin)')
    args = parser.parse_args()

    value = args.value
    target_unit = args.to

    if target_unit == 'f':
        result = converters.convert_c_to_f(value)
    elif target_unit == 'k':
        result = converters.convert_c_to_k(value)
    elif target_unit == 'c':
        result = value
    else:
        parser.error(f"Invalid target unit: {target_unit}. Use c, f, or k.")

    print(result)

if __name__ == '__main__':
    main()

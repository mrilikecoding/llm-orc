import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature')
    parser.add_argument('value', type=float, help='The temperature value')
    parser.add_argument('source_unit', choices=['C', 'F'], help='Source unit (C or F)')
    parser.add_argument('target_unit', choices=['C', 'F', 'K'], help='Target unit (C, F, or K)')
    args = parser.parse_args()

    value = args.value
    source = args.source_unit
    target = args.target_unit

    if source == 'C':
        if target == 'F':
            result = converters.celsius_to_fahrenheit(value)
        elif target == 'K':
            result = converters.celsius_to_kelvin(value)
        else:
            parser.error(f"Invalid target unit: {target}. Must be F or K.")
    elif source == 'F':
        if target == 'C':
            result = converters.fahrenheit_to_celsius(value)
        else:
            parser.error(f"Invalid target unit: {target}. Must be C.")
    else:
        parser.error(f"Invalid source unit: {source}. Must be C or F.")

    print(result)

if __name__ == '__main__':
    main()

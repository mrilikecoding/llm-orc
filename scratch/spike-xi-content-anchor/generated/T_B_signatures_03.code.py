import sys
import converters

def main():
    if len(sys.argv) != 3:
        print("Usage: cli.py <value> <target_unit>")
        return

    try:
        value = float(sys.argv[1])
        target_unit = sys.argv[2].upper()
    except ValueError:
        print("Invalid value. Please provide a numeric value.")
        return

    if target_unit == 'F':
        result = converters.celsius_to_fahrenheit(value)
    elif target_unit == 'C':
        result = converters.fahrenheit_to_celsius(value)
    elif target_unit == 'K':
        result = converters.celsius_to_kelvin(value)
    else:
        print("Invalid target unit. Use F, C, or K.")
        return

    print(result)

if __name__ == "__main__":
    main()

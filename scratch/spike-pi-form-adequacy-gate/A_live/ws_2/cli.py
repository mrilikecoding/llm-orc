import sys
import converters

def main():
    if len(sys.argv) != 4:
        print("Usage: script.py <value> <from_unit> <to_unit>")
        return

    value = float(sys.argv[1])
    from_unit = sys.argv[2].strip().lower()
    to_unit = sys.argv[3].strip().lower()

    if from_unit == 'celsius' and to_unit == 'fahrenheit':
        result = converters.celsius_to_fahrenheit(value)
    elif from_unit == 'fahrenheit' and to_unit == 'celsius':
        result = converters.fahrenheit_to_celsius(value)
    elif from_unit == 'celsius' and to_unit == 'kelvin':
        result = converters.celsius_to_kelvin(value)
    else:
        print("Unsupported conversion")
        return

    print(f"{value} {from_unit} is equal to {result} {to_unit}")

if __name__ == "__main__":
    main()
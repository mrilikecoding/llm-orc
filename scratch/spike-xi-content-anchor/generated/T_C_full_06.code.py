import sys
import converters

def main():
    if len(sys.argv) != 4:
        print("Usage: cli.py <value> <from_unit> <to_unit>")
        return

    try:
        value = float(sys.argv[1])
    except ValueError:
        print("Invalid value. Please provide a numeric value.")
        return

    from_unit = sys.argv[2].strip().capitalize()
    to_unit = sys.argv[3].strip().capitalize()

    if from_unit == 'Celsius' and to_unit == 'Fahrenheit':
        result = converters.celsius_to_fahrenheit(value)
    elif from_unit == 'Fahrenheit' and to_unit == 'Celsius':
        result = converters.fahrenheit_to_celsius(value)
    elif from_unit == 'Celsius' and to_unit == 'Kelvin':
        result = converters.celsius_to_kelvin(value)
    else:
        print("Unsupported conversion. Use Celsius, Fahrenheit, or Kelvin.")
        return

    print(result)

if __name__ == "__main__":
    main()

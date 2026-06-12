import sys
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    if len(sys.argv) != 4:
        print("Usage: python temp_conv.py <value> <from_unit> <to_unit>")
        return

    value = float(sys.argv[1])
    from_unit = sys.argv[2].capitalize()
    to_unit = sys.argv[3].capitalize()

    if from_unit == to_unit:
        print(f"{value} {from_unit} is equal to {value} {to_unit}")
        return

    if from_unit == 'Celsius' and to_unit == 'Fahrenheit':
        result = celsius_to_fahrenheit(value)
    elif from_unit == 'Celsius' and to_unit == 'Kelvin':
        result = celsius_to_kelvin(value)
    elif from_unit == 'Fahrenheit' and to_unit == 'Celsius':
        result = fahrenheit_to_celsius(value)
    elif from_unit == 'Fahrenheit' and to_unit == 'Kelvin':
        celsius = fahrenheit_to_celsius(value)
        result = celsius_to_kelvin(celsius)
    elif from_unit == 'Kelvin' and to_unit == 'Celsius':
        result = value - 273.15
    elif from_unit == 'Kelvin' and to_unit == 'Fahrenheit':
        celsius = value - 273.15
        result = celsius_to_fahrenheit(celsius)
    else:
        print("Unsupported unit conversion")
        return

    print(f"{value} {from_unit} is equal to {result} {to_unit}")

if __name__ == "__main__":
    main()
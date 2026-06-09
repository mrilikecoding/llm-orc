import sys
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius

def main():
    if len(sys.argv) != 3:
        print("Usage: cli.py <value> <target_unit>")
        print("Example: cli.py 25 F  # Converts 25°C to Fahrenheit")
        print("        cli.py 77 C  # Converts 77°F to Celsius")
        return

    try:
        value = float(sys.argv[1])
        target_unit = sys.argv[2].upper()
    except ValueError:
        print("Error: Value must be a number.")
        return

    if target_unit == 'F':
        result = celsius_to_fahrenheit(value)
        print(f"{value}°C = {result:.2f}°F")
    elif target_unit == 'C':
        result = fahrenheit_to_celsius(value)
        print(f"{value}°F = {result:.2f}°C")
    else:
        print("Invalid target unit. Use 'C' or 'F'.")

if __name__ == "__main__":
    main()

import sys
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    if len(sys.argv) != 4:
        print("Usage: python cli.py <value> <from_unit> <to_unit>")
        sys.exit(1)
    
    value_str = sys.argv[1]
    try:
        value = float(value_str)
    except ValueError:
        print("Error: Value must be a number.")
        sys.exit(1)
    
    from_unit = sys.argv[2]
    to_unit = sys.argv[3]
    
    allowed_units = {'C', 'F', 'K'}
    if from_unit not in allowed_units or to_unit not in allowed_units:
        print("Error: Unknown units. Supported units: C, F, K.")
        sys.exit(1)
    
    conversion_map = {
        ('C', 'F'): celsius_to_fahrenheit,
        ('F', 'C'): fahrenheit_to_celsius,
        ('C', 'K'): celsius_to_kelvin
    }
    
    if (from_unit, to_unit) not in conversion_map:
        print(f"Error: Conversion from {from_unit} to {to_unit} is not supported.")
        sys.exit(1)
    
    converter = conversion_map[(from_unit, to_unit)]
    result = converter(value)
    print(f"{result:.2f}")

if __name__ == "__main__":
    main()
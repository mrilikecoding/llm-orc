import sys
import converters

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: cli.py <value> <source_unit> <target_unit>")
        sys.exit(1)
    
    value = float(sys.argv[1])
    source = sys.argv[2].upper()
    target = sys.argv[3].upper()
    
    if source not in ('C', 'F', 'K') or target not in ('C', 'F', 'K'):
        print("Invalid unit. Use C, F, or K.")
        sys.exit(1)
    
    if source == 'C' and target == 'F':
        print(converters.celsius_to_fahrenheit(value))
    elif source == 'F' and target == 'C':
        print(converters.fahrenheit_to_celsius(value))
    elif source == 'C' and target == 'K':
        print(converters.celsius_to_kelvin(value))
    else:
        print("Unsupported conversion.")
        sys.exit(1)

import sys
import converters

def main():
    if len(sys.argv) != 4:
        print("Usage: cli.py <value> <from_unit> <to_unit>")
        return
    try:
        value = float(sys.argv[1])
        from_unit = sys.argv[2].upper()
        to_unit = sys.argv[3].upper()
        result = converters.convert(value, from_unit, to_unit)
        print(f"{value} {from_unit} is equal to {result} {to_unit}")
    except ValueError:
        print("Error: Value must be a number.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

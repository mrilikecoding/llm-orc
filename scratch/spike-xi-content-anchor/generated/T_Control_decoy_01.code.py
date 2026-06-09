import converters
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: python cli.py <value> <from_unit> <to_unit>")
        sys.exit(1)
    value = float(sys.argv[1])
    from_unit = sys.argv[2]
    to_unit = sys.argv[3]
    result = converters.convert_temperature(value, from_unit, to_unit)
    print(result)

if __name__ == "__main__":
    main()

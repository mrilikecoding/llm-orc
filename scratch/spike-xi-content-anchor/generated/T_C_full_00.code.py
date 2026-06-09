import sys
import converters

def main():
    if len(sys.argv) != 3:
        print("Usage: python cli.py <value> <conversion_function>")
        return

    try:
        value = float(sys.argv[1])
    except ValueError:
        print("Invalid value provided.")
        return

    func_name = sys.argv[2]
    func = getattr(converters, func_name, None)
    if not func:
        print(f"Unknown function: {func_name}")
        return

    result = func(value)
    print(result)

if __name__ == "__main__":
    main()

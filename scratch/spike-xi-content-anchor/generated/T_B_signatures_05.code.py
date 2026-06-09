import sys
import converters

def main():
    if len(sys.argv) != 3:
        return  # Assume valid input as per problem constraints

    value = float(sys.argv[1])
    target = sys.argv[2].upper()

    if target == 'F':
        result = converters.celsius_to_fahrenheit(value)
    elif target == 'C':
        result = converters.fahrenheit_to_celsius(value)
    elif target == 'K':
        result = converters.celsius_to_kelvin(value)
    else:
        return  # Invalid target unit

    print(result)

if __name__ == '__main__':
    main()

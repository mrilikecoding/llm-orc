import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('from_unit', type=str, help='The original unit (e.g., Celsius, Fahrenheit, Kelvin).')
    parser.add_argument('to_unit', type=str, help='The target unit (e.g., Celsius, Fahrenheit, Kelvin).')
    args = parser.parse_args()
    
    result = converters.convert_temperature(args.value, args.from_unit, args.to_unit)
    print(result)

if __name__ == '__main__':
    main()

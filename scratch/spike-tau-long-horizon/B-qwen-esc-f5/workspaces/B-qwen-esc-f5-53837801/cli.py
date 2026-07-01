import argparse
from converters import celsius_to_fahrenheit, fahrenheit_to_celsius, celsius_to_kelvin

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius, Fahrenheit, and Kelvin.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('source', choices=['celsius', 'fahrenheit', 'kelvin'], help='The source unit.')
    parser.add_argument('target', choices=['celsius', 'fahrenheit', 'kelvin'], help='The target unit.')
    
    args = parser.parse_args()
    
    value = args.value
    source = args.source
    target = args.target
    
    if source == target:
        print(f"{value} {source} is equal to {value} {target}.")
        return
    
    if source == 'celsius' and target == 'fahrenheit':
        result = celsius_to_fahrenheit(value)
    elif source == 'fahrenheit' and target == 'celsius':
        result = fahrenheit_to_celsius(value)
    elif source == 'celsius' and target == 'kelvin':
        result = celsius_to_kelvin(value)
    else:
        print(f"Error: Conversion from {source} to {target} is not supported.")
        return
    
    print(f"{value} {source} is equal to {result} {target}.")

if __name__ == '__main__':
    main()
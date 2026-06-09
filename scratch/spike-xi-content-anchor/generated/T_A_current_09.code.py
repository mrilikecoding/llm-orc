import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between Celsius, Fahrenheit, and Kelvin.')
    parser.add_argument('value', type=float, help='The temperature value')
    parser.add_argument('from_unit', choices=['C', 'F', 'K'], help='Source unit (C, F, K)')
    parser.add_argument('to_unit', choices=['C', 'F', 'K'], help='Target unit (C, F, K)')
    args = parser.parse_args()

    if args.from_unit == 'C':
        if args.to_unit == 'F':
            result = converters.c_to_f(args.value)
        elif args.to_unit == 'K':
            result = converters.c_to_k(args.value)
    elif args.from_unit == 'F':
        if args.to_unit == 'C':
            result = converters.f_to_c(args.value)
        elif args.to_unit == 'K':
            result = converters.f_to_k(args.value)
    elif args.from_unit == 'K':
        if args.to_unit == 'C':
            result = converters.k_to_c(args.value)
        elif args.to_unit == 'F':
            result = converters.k_to_f(args.value)

    print(result)

if __name__ == '__main__':
    main()

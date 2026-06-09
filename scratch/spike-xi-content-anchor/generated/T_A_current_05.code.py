import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units.')
    parser.add_argument('value', type=float, help='The temperature value to convert.')
    parser.add_argument('source_unit', choices=['F', 'C', 'K'], help='Source unit (F, C, K)')
    parser.add_argument('target_unit', choices=['F', 'C', 'K'], help='Target unit (F, C, K)')
    args = parser.parse_args()

    if args.source_unit == 'F' and args.target_unit == 'C':
        result = converters.f_to_c(args.value)
    elif args.source_unit == 'C' and args.target_unit == 'F':
        result = converters.c_to_f(args.value)
    elif args.source_unit == 'C' and args.target_unit == 'K':
        result = converters.c_to_k(args.value)
    elif args.source_unit == 'K' and args.target_unit == 'C':
        result = converters.k_to_c(args.value)
    else:
        raise ValueError("Unsupported conversion type")

    print(f"{args.value} {args.source_unit} is equal to {result} {args.target_unit}")

if __name__ == '__main__':
    main()

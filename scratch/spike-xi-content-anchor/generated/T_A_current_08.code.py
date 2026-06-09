import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature between units')
    parser.add_argument('value', type=float, help='The temperature value to convert')
    parser.add_argument('source_unit', choices=['C', 'F', 'K'], help='Source unit (C, F, K)')
    parser.add_argument('target_unit', choices=['C', 'F', 'K'], help='Target unit (C, F, K)')
    args = parser.parse_args()

    conversion_functions = {
        ('C', 'F'): converters.c_to_f,
        ('F', 'C'): converters.f_to_c,
        ('C', 'K'): converters.c_to_k,
        ('K', 'C'): converters.k_to_c,
        ('F', 'K'): converters.f_to_k,
        ('K', 'F'): converters.k_to_f,
    }

    func = conversion_functions.get((args.source_unit, args.target_unit))
    if not func:
        print("Unsupported conversion")
        return

    result = func(args.value)
    print(result)

if __name__ == '__main__':
    main()

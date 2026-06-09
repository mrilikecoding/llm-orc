import argparse
import converters

def main():
    parser = argparse.ArgumentParser(description='Convert temperature')
    parser.add_argument('value', type=float, help='Temperature value')
    parser.add_argument('from_unit', choices=['C', 'F', 'K'], help='From unit (C, F, K)')
    parser.add_argument('to_unit', choices=['C', 'F', 'K'], help='To unit (C, F, K)')
    args = parser.parse,args
    result = converters.convert_temperature(args.value, args.from_unit, args.to_unit)
    print(result)

if __name__ == '__main__':
    main()
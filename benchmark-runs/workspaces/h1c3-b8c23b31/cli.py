import argparse

parser = argparse.ArgumentParser(description='Process a number with double or halve.')
parser.add_argument('number', type=int, help='The numeric argument')
parser.add_argument('--double', action='store_true', help='Double the number')
parser.add_argument('--halve', action='store_true', help='Halve the number')

args = parser.parse_args()

if args.double and args.halve:
    print("Error: Cannot use both --double and --halve flags together.")
    exit(1)

result = args.number
if args.double:
    result *= 2
elif args.halve:
    result //= 2

print(result)
import argparse
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate

def main() -> None:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("expression")
    args = arg_parser.parse_args()
    tokens = tokenize(args.expression)
    ast = parse(tokens)
    result = evaluate(ast)
    print(result)
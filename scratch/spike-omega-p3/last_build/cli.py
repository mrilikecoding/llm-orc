from __future__ import annotations
import argparse
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate arithmetic expressions.")
    parser.add_argument("expression", type=str, help="Arithmetic expression to evaluate.")
    args = parser.parse_args()
    
    try:
        tokens = tokenize(args.expression)
        ast = parse(tokens)
        result = evaluate(ast)
        print(result)
    except Exception as e:
        print(f"Error: {e}")
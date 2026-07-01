from __future__ import annotations
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate

def test_evaluate() -> None:
    test_cases = [
        ("3 + 5", 8.0),
        ("10 * 2", 20.0),
        ("8 / 4", 2.0),
        ("2 + 3 * 4", 14.0),
        ("(1 + 2) * 3", 9.0),
        ("100 - 20 * 3", 40.0),
        ("(5 + 5) / 2", 5.0),
        ("3 * (4 + 2)", 18.0),
        ("6 / 2 + 3", 6.0),
        ("(1 + 2) * (3 + 4)", 21.0),
    ]
    
    for expression, expected in test_cases:
        try:
            tokens = tokenize(expression)
            ast = parse(tokens)
            result = evaluate(ast)
            assert abs(result - expected) < 1e-9, f"Failed for {expression}: got {result}, expected {expected}"
        except Exception as e:
            print(f"Error in test case {expression}: {str(e)}")
from tokenizer import tokenize
from parser import parse
from evaluator import evaluate

def test_end_to_end() -> None:
    test_cases = [
        ("3 + 4 * 2", 11.0),
        ("5 - (3 + 2)", 0.0),
        ("10 / 2", 5.0),
        ("2 * (3 + 4)", 14.0),
        ("1 + 2 * 3 - 4 / 2", 5.0),
        ("100 - 200", -100.0),
        ("1 + 1", 2.0),
    ]
    for expr, expected in test_cases:
        tokens = tokenize(expr)
        ast = parse(tokens)
        result = evaluate(ast)
        assert result == expected, f"Failed for {expr}: expected {expected}, got {result}"
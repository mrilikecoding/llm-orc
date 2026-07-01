from __future__ import annotations
from parser import Number, BinaryOp

def evaluate(ast: object) -> float:
    if isinstance(ast, Number):
        return ast.value
    elif isinstance(ast, BinaryOp):
        left_val = evaluate(ast.left)
        right_val = evaluate(ast.right)
        if ast.op == '+':
            return left_val + right_val
        elif ast.op == '-':
            return left_val - right_val
        elif ast.op == '*':
            return left_val * right_val
        elif ast.op == '/':
            return left_val / right_val
    raise ValueError("Unknown operation")
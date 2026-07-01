from parser import Number, BinOp
from typing import Union

def evaluate(node: Union[Number, BinOp]) -> float:
    if isinstance(node, Number):
        return node.value
    left_val = evaluate(node.left)
    right_val = evaluate(node.right)
    if node.op == '+':
        return left_val + right_val
    elif node.op == '-':
        return left_val - right_val
    elif node.op == '*':
        return left_val * right_val
    elif node.op == '/':
        return left_val / right_val
    raise ValueError(f"Unknown operator: {node.op}")
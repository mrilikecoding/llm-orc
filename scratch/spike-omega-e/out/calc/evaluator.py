from __future__ import annotations
from models import tokens
from dataclasses import dataclass

@dataclass
class operations:
    add: str
    subtract: str
    multiply: str
    divide: str

def evaluate(ast: dict[str, str]) -> int:
    if 'type' in ast:
        left = evaluate(ast['left'])
        right = evaluate(ast['right'])
        if ast['type'] == 'add':
            return left + right
        elif ast['type'] == 'subtract':
            return left - right
        elif ast['type'] == 'multiply':
            return left * right
        elif ast['type'] == 'divide':
            return left // right
    else:
        return int(ast)
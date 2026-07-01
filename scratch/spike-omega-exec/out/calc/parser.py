from dataclasses import dataclass
from tokenizer import Token
from typing import Union

@dataclass
class Number:
    value: float

@dataclass
class BinOp:
    left: Union[Number, BinOp]
    op: str
    right: Union[Number, BinOp]

def parse(tokens: list[Token]) -> Union[Number, BinOp]:
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def consume():
        nonlocal pos
        t = tokens[pos]
        pos += 1
        return t

    def parse_primary():
        t = consume()
        if t.type == 'NUMBER':
            return Number(float(t.value))
        if t.value == '(':
            node = parse_expr()
            if peek() and peek().value == ')':
                consume()
            return node
        raise ValueError(f"Unexpected token: {t.value}")

    def parse_term():
        left = parse_primary()
        while peek() and peek().value in ('*', '/'):
            op = consume().value
            right = parse_primary()
            left = BinOp(left, op, right)
        return left

    def parse_expr():
        left = parse_term()
        while peek() and peek().value in ('+', '-'):
            op = consume().value
            right = parse_term()
            left = BinOp(left, op, right)
        return left

    result = parse_expr()
    if pos < len(tokens):
        raise ValueError("Unexpected tokens remaining")
    return result
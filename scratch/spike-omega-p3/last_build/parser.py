from __future__ import annotations
from dataclasses import dataclass
from tokenizer import Token

@dataclass
class Number:
    value: float

@dataclass
class BinaryOp:
    op: str
    left: object
    right: object

def parse(tokens: list[Token]) -> object:
    return parse_expression(tokens)

def parse_expression(tokens: list[Token]) -> object:
    left = parse_term(tokens)
    while tokens and tokens[0].kind in ('+', '-'):
        op = tokens[0].kind
        tokens.pop(0)
        right = parse_term(tokens)
        left = BinaryOp(op, left, right)
    return left

def parse_term(tokens: list[Token]) -> object:
    left = parse_factor(tokens)
    while tokens and tokens[0].kind in ('*', '/'):
        op = tokens[0].kind
        tokens.pop(0)
        right = parse_factor(tokens)
        left = BinaryOp(op, left, right)
    return left

def parse_factor(tokens: list[Token]) -> object:
    if tokens[0].kind == 'NUMBER':
        token = tokens.pop(0)
        return Number(token.value)
    elif tokens[0].kind == '(':
        tokens.pop(0)
        expr = parse_expression(tokens)
        if tokens and tokens[0].kind == ')':
            tokens.pop(0)
        return expr
    else:
        raise ValueError("Unexpected token")
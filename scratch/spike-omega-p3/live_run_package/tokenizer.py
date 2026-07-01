from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Token:
    kind: str
    value: str | float

def tokenize(expression: str) -> list[Token]:
    tokens = []
    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isspace():
            i += 1
            continue
        if char in "+-*/()":
            tokens.append(Token("operator", char))
            i += 1
        elif char.isdigit() or (char == '-' and i + 1 < len(expression) and expression[i+1].isdigit()):
            num = ""
            while i < len(expression) and (expression[i].isdigit() or expression[i] == '.'):
                num += expression[i]
                i += 1
            tokens.append(Token("number", float(num)))
        elif char == '(' or char == ')':
            tokens.append(Token("paren", char))
            i += 1
        else:
            i += 1
    return tokens
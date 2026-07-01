from __future__ import annotations
from models import tokens
from dataclasses import dataclass

@dataclass
class operations:
    add: str
    subtract: str
    multiply: str
    divide: str

def parse(tokens: list[str]) -> dict[str, str]:
    if len(tokens) != 3:
        raise ValueError("Invalid number of tokens")
    operator = tokens[1]
    left = tokens[0]
    right = tokens[2]
    return {
        "type": operator,
        "left": left,
        "right": right
    }
from __future__ import annotations
from models import tokens
import re

@dataclass
class tokens:
    operand: str
    op: str
    operand: str

def tokenize(expression: str) -> list[str]:
    tokens = re.findall(r'\d+|\+|\-|\*|\/', expression)
    return tokens
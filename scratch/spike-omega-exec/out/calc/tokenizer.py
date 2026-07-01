from dataclasses import dataclass

@dataclass
class Token:
    type: str
    value: str

def tokenize(expr: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    while i < len(expr):
        char = expr[i]
        if char.isspace():
            i += 1
        elif char.isdigit() or char == '.':
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(Token(type='NUMBER', value=expr[i:j]))
            i = j
        elif char in '+-*/()':
            tokens.append(Token(type=char, value=char))
            i += 1
        else:
            raise ValueError(f"Invalid character: {char}")
    return tokens
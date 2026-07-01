from parser import parse_int
from validators import in_range

def run(text, low, high):
    parsed = parse_int(text)
    if in_range(parsed, low, high):
        return parsed
    else:
        raise ValueError("Value out of range")
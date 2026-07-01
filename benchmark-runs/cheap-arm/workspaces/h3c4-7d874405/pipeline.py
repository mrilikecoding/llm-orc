from parser import parse_int
from validators import in_range

def run(text, low, high):
    value = parse_int(text)
    if in_range(value, low, high):
        return value